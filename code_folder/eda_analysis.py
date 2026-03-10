from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import psycopg
from psycopg import sql
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

import transform_pipeline as tp


@dataclass
class EncodingState:
    freq_maps: dict[str, dict[str, float]] | None = None
    int_maps: dict[str, dict[str, int]] | None = None
    counts: dict[str, Counter[str]] | None = None
    train_rows: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "EDA + modeling workflow: load transformed source table, encode categorical "
            "features, train LightGBM, and rank features by gain importance."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--pg-dsn",
        default=os.getenv(
            "PG_DSN",
            "postgresql://local_user:Abc$12345@docker-lcpostgres:5432/migration_db_test",
        ),
    )
    parser.add_argument(
        "--source-table",
        default=os.getenv("SOURCE_TABLE", "mlops.ml_features_full_part_test"),
    )
    parser.add_argument(
        "--source-partition-like",
        default=os.getenv("SOURCE_PARTITION_LIKE", "%"),
        help="SQL LIKE filter for partition_file, e.g. 'day_2%%' or 'day_%%'.",
    )
    parser.add_argument(
        "--train-mode",
        choices=["in_memory", "chunked"],
        default=os.getenv("TRAIN_MODE", "in_memory"),
        help="Use 'chunked' for full-table streaming training with bounded memory.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=int(os.getenv("MAX_ROWS", "1000000")),
        help="Maximum rows to read. Use 0 or negative for no LIMIT.",
    )
    parser.add_argument(
        "--fetch-size",
        type=int,
        default=int(os.getenv("FETCH_SIZE", "200000")),
        help="DB batch size for server-side cursor fetchmany().",
    )
    parser.add_argument(
        "--cat-encoding",
        choices=["frequency", "integer"],
        default=os.getenv("CAT_ENCODING", "frequency"),
        help="Categorical encoding strategy.",
    )
    parser.add_argument(
        "--train-size",
        type=float,
        default=float(os.getenv("TRAIN_SIZE", "0.6")),
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=float(os.getenv("VAL_SIZE", "0.20")),
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=float(os.getenv("TEST_SIZE", "0.20")),
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=int(os.getenv("RANDOM_STATE", "42")),
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=int(os.getenv("N_ESTIMATORS", "500")),
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=float(os.getenv("LEARNING_RATE", "0.05")),
    )
    parser.add_argument(
        "--num-leaves",
        type=int,
        default=int(os.getenv("NUM_LEAVES", "63")),
    )
    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=int(os.getenv("EARLY_STOPPING_ROUNDS", "50")),
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=int(os.getenv("N_JOBS", "-1")),
    )
    parser.add_argument(
        "--enable-hyperopt",
        type=int,
        default=int(os.getenv("ENABLE_HYPEROPT", "0")),
        choices=[0, 1],
        help="Run random-search hyperparameter tuning (in_memory mode only).",
    )
    parser.add_argument(
        "--hyperopt-trials",
        type=int,
        default=int(os.getenv("HYPEROPT_TRIALS", "12")),
        help="Number of random-search trials when hyperopt is enabled.",
    )
    parser.add_argument(
        "--chunk-boost-rounds",
        type=int,
        default=int(os.getenv("CHUNK_BOOST_ROUNDS", "20")),
        help="Boosting rounds to add per training chunk when --train-mode=chunked.",
    )
    parser.add_argument(
        "--eval-max-rows",
        type=int,
        default=int(os.getenv("EVAL_MAX_ROWS", "300000")),
        help="Max rows kept for val/test evaluation buffers in chunked mode.",
    )
    parser.add_argument(
        "--log-every-chunks",
        type=int,
        default=int(os.getenv("LOG_EVERY_CHUNKS", "5")),
    )
    parser.add_argument(
        "--encoding-debug-top-k",
        type=int,
        default=int(os.getenv("ENCODING_DEBUG_TOP_K", "5")),
        help="Show top-k categories per categorical feature in encoding report.",
    )
    parser.add_argument(
        "--print-encoding-summary",
        type=int,
        default=int(os.getenv("PRINT_ENCODING_SUMMARY", "1")),
        choices=[0, 1],
        help="Print a concise encoding summary to stdout.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=int(os.getenv("TOP_N", "40")),
    )
    parser.add_argument(
        "--importance-csv",
        default=os.getenv(
            "IMPORTANCE_CSV", "/app/artifacts/eda_lgbm_feature_importance_gain.csv"
        ),
    )
    parser.add_argument(
        "--metrics-json",
        default=os.getenv("METRICS_JSON", "/app/artifacts/eda_lgbm_metrics.json"),
    )
    parser.add_argument(
        "--encoding-report-json",
        default=os.getenv(
            "ENCODING_REPORT_JSON", "/app/artifacts/eda_encoding_report.json"
        ),
    )
    parser.add_argument(
        "--tuning-results-csv",
        default=os.getenv("TUNING_RESULTS_CSV", "/app/artifacts/eda_lgbm_tuning_results.csv"),
    )
    return parser.parse_args()


def validate_split_sizes(train_size: float, val_size: float, test_size: float) -> None:
    total = train_size + val_size + test_size
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(
            f"train_size + val_size + test_size must equal 1.0, got {total:.8f}"
        )


def split_3way(
    df: pd.DataFrame,
    train_size: float,
    val_size: float,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validate_split_sizes(train_size, val_size, test_size)

    y_all = df[tp.LABEL_COL].astype(int)
    strat_all = y_all if y_all.nunique(dropna=False) > 1 else None

    train_df, rem_df = train_test_split(
        df,
        train_size=train_size,
        random_state=random_state,
        shuffle=True,
        stratify=strat_all,
    )

    rem_test_ratio = test_size / (val_size + test_size)
    y_rem = rem_df[tp.LABEL_COL].astype(int)
    strat_rem = y_rem if y_rem.nunique(dropna=False) > 1 else None

    val_df, test_df = train_test_split(
        rem_df,
        test_size=rem_test_ratio,
        random_state=random_state,
        shuffle=True,
        stratify=strat_rem,
    )
    return train_df, val_df, test_df


def _select_stmt(source: tp.DbTable, with_limit: bool) -> sql.SQL:
    cols = [tp.LABEL_COL] + tp.INT_COLS + tp.CAT_COLS + ["partition_file"]
    if with_limit:
        return sql.SQL("SELECT {} FROM {} WHERE partition_file LIKE %s LIMIT %s").format(
            sql.SQL(", ").join(map(sql.Identifier, cols)),
            sql.Identifier(source.schema, source.table),
        )
    return sql.SQL("SELECT {} FROM {} WHERE partition_file LIKE %s").format(
        sql.SQL(", ").join(map(sql.Identifier, cols)),
        sql.Identifier(source.schema, source.table),
    )


def iter_source_chunks(
    dsn: str,
    source_table: str,
    source_partition_like: str,
    max_rows: int,
    fetch_size: int,
    cursor_name: str,
):
    source = tp.split_qualified(source_table)
    with_limit = max_rows > 0
    q = _select_stmt(source, with_limit=with_limit)

    loaded_raw = 0
    with psycopg.connect(dsn) as conn:
        with conn.cursor(name=cursor_name, withhold=True) as cur:
            if with_limit:
                cur.execute(q, (source_partition_like, max_rows))
            else:
                cur.execute(q, (source_partition_like,))

            while True:
                if with_limit:
                    remaining = max_rows - loaded_raw
                    if remaining <= 0:
                        break
                    size = min(fetch_size, remaining)
                else:
                    size = fetch_size

                rows = cur.fetchmany(size)
                if not rows:
                    break

                loaded_raw += len(rows)
                chunk = pd.DataFrame(
                    rows,
                    columns=[tp.LABEL_COL] + tp.INT_COLS + tp.CAT_COLS + ["partition_file"],
                )
                if tp.LABEL_COL not in chunk.columns:
                    continue
                chunk = chunk.dropna(subset=[tp.LABEL_COL]).copy()
                if chunk.empty:
                    continue
                chunk[tp.LABEL_COL] = chunk[tp.LABEL_COL].astype(int)
                yield chunk, loaded_raw


def load_source_df(
    dsn: str,
    source_table: str,
    source_partition_like: str,
    max_rows: int,
    fetch_size: int,
) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    loaded = 0
    for chunk, loaded_raw in iter_source_chunks(
        dsn=dsn,
        source_table=source_table,
        source_partition_like=source_partition_like,
        max_rows=max_rows,
        fetch_size=fetch_size,
        cursor_name="eda_src_inmem",
    ):
        chunks.append(chunk)
        loaded += len(chunk)
        print(f"[load] loaded_rows={loaded} raw_rows_seen={loaded_raw}")

    if not chunks:
        raise ValueError("No rows returned from source table for given filters.")

    return pd.concat(chunks, ignore_index=True)


def deterministic_split_masks(
    df: pd.DataFrame,
    train_size: float,
    val_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Hash-based split is stable across passes/chunks without retaining all row ids.
    split_cols = ["partition_file", tp.LABEL_COL, tp.INT_COLS[0], tp.CAT_COLS[0]]
    hashed = pd.util.hash_pandas_object(df[split_cols], index=False).to_numpy(dtype=np.uint64)
    mixed = hashed ^ np.uint64(random_state)
    u = (mixed % np.uint64(10_000_019)).astype(np.float64) / 10_000_019.0

    train_mask = u < train_size
    val_mask = (u >= train_size) & (u < (train_size + val_size))
    test_mask = ~(train_mask | val_mask)
    return train_mask, val_mask, test_mask


def fit_encoding_state(train_df: pd.DataFrame, cat_encoding: str) -> EncodingState:
    counts: dict[str, Counter[str]] = {}
    for c in tp.CAT_COLS:
        vc = (
            train_df[c]
            .fillna(tp.MISSING_TOKEN)
            .astype(str)
            .value_counts(dropna=False)
        )
        counts[c] = Counter(vc.to_dict())

    if cat_encoding == "frequency":
        freq_maps: dict[str, dict[str, float]] = {}
        denom = float(max(len(train_df), 1))
        for c in tp.CAT_COLS:
            freq_maps[c] = {k: (v / denom) for k, v in counts[c].items()}
        return EncodingState(freq_maps=freq_maps, counts=counts, train_rows=len(train_df))

    int_maps: dict[str, dict[str, int]] = {}
    for c in tp.CAT_COLS:
        ordered = [k for k, _ in counts[c].most_common()]
        int_maps[c] = {v: i + 1 for i, v in enumerate(ordered)}
    return EncodingState(int_maps=int_maps, counts=counts, train_rows=len(train_df))


def encode_frame_with_state(
    frame: pd.DataFrame,
    state: EncodingState,
    cat_encoding: str,
) -> pd.DataFrame:
    out = frame.copy()
    if cat_encoding == "frequency":
        if state.freq_maps is None:
            raise ValueError("Frequency maps are missing in encoding state.")
        for c in tp.CAT_COLS:
            out[c] = (
                out[c]
                .fillna(tp.MISSING_TOKEN)
                .astype(str)
                .map(state.freq_maps[c])
                .fillna(0.0)
            )
    else:
        if state.int_maps is None:
            raise ValueError("Integer maps are missing in encoding state.")
        for c in tp.CAT_COLS:
            out[c] = (
                out[c]
                .fillna(tp.MISSING_TOKEN)
                .astype(str)
                .map(state.int_maps[c])
                .fillna(0)
                .astype(np.int32)
            )
    return out


def count_unknown_categories(
    frame: pd.DataFrame,
    state: EncodingState,
    cat_encoding: str,
) -> dict[str, int]:
    out: dict[str, int] = {}
    if cat_encoding == "frequency":
        if state.freq_maps is None:
            raise ValueError("Frequency maps are missing in encoding state.")
        for c in tp.CAT_COLS:
            values = frame[c].fillna(tp.MISSING_TOKEN).astype(str)
            out[c] = int((~values.isin(state.freq_maps[c])).sum())
    else:
        if state.int_maps is None:
            raise ValueError("Integer maps are missing in encoding state.")
        for c in tp.CAT_COLS:
            values = frame[c].fillna(tp.MISSING_TOKEN).astype(str)
            out[c] = int((~values.isin(state.int_maps[c])).sum())
    return out


def _top_categories_for_report(
    counts: Counter[str],
    top_k: int,
    train_rows: int,
) -> list[dict[str, Any]]:
    top_items: list[dict[str, Any]] = []
    denom = float(max(train_rows, 1))
    for value, count in counts.most_common(max(1, top_k)):
        top_items.append(
            {
                "value": str(value),
                "count": int(count),
                "train_pct": (float(count) / denom) * 100.0,
            }
        )
    return top_items


def build_encoding_report(
    state: EncodingState,
    cat_encoding: str,
    top_k: int,
    unknown_stats: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    features: list[dict[str, Any]] = []
    for c in tp.CAT_COLS:
        if cat_encoding == "frequency":
            mapping_size = len(state.freq_maps[c]) if state.freq_maps else 0
        else:
            mapping_size = len(state.int_maps[c]) if state.int_maps else 0
        counts = state.counts[c] if state.counts else Counter()
        features.append(
            {
                "feature": c,
                "mapping_size": int(mapping_size),
                "train_unique_categories": int(len(counts)),
                "top_categories": _top_categories_for_report(
                    counts=counts,
                    top_k=top_k,
                    train_rows=state.train_rows,
                ),
            }
        )
    return {
        "encoding": cat_encoding,
        "train_rows_for_mapping": int(state.train_rows),
        "feature_summary": features,
        "split_unknowns": unknown_stats or {},
    }


def print_encoding_summary(report: dict[str, Any]) -> None:
    print(
        "[encoding] strategy="
        f"{report['encoding']} train_rows_for_mapping={report['train_rows_for_mapping']}"
    )
    for item in report.get("feature_summary", []):
        print(
            "[encoding] "
            f"{item['feature']} mapping_size={item['mapping_size']} "
            f"train_unique={item['train_unique_categories']}"
        )


def safe_binary_metrics(y_true: np.ndarray | pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    out: dict[str, float] = {}
    y_arr = np.asarray(y_true)
    if y_arr.size == 0:
        out["auc"] = float("nan")
        out["logloss"] = float("nan")
        return out

    if np.unique(y_arr).size < 2:
        out["auc"] = float("nan")
    else:
        out["auc"] = float(roc_auc_score(y_arr, y_pred))
    out["logloss"] = float(log_loss(y_arr, y_pred, labels=[0, 1]))
    return out


def ensure_numeric_features(df: pd.DataFrame) -> None:
    features = tp.INT_COLS + tp.CAT_COLS
    df[features] = df[features].apply(pd.to_numeric, errors="coerce")
    df[features] = df[features].fillna(0.0)


def _build_base_lgbm_params(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "objective": "binary",
        "n_estimators": int(args.n_estimators),
        "learning_rate": float(args.learning_rate),
        "num_leaves": int(args.num_leaves),
        "random_state": int(args.random_state),
        "n_jobs": int(args.n_jobs),
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }


def _sample_hyperopt_params(rng: np.random.Generator, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "n_estimators": int(rng.integers(250, max(int(args.n_estimators) * 2, 251))),
        "learning_rate": float(np.exp(rng.uniform(np.log(0.01), np.log(0.2)))),
        "num_leaves": int(rng.integers(31, 256)),
        "min_child_samples": int(rng.integers(20, 301)),
        "subsample": float(rng.uniform(0.6, 1.0)),
        "colsample_bytree": float(rng.uniform(0.6, 1.0)),
        "reg_alpha": float(rng.uniform(0.0, 2.0)),
        "reg_lambda": float(rng.uniform(0.0, 5.0)),
        "max_depth": int(rng.choice(np.array([-1, 6, 8, 10, 12]))),
    }


def _fit_lgbm_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    params: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[lgb.LGBMClassifier, dict[str, float | int]]:
    model = lgb.LGBMClassifier(**params)

    callbacks = []
    if args.early_stopping_rounds > 0 and len(x_val) > 0:
        callbacks.append(lgb.early_stopping(args.early_stopping_rounds, verbose=False))

    eval_set = [(x_val, y_val)] if len(x_val) > 0 else None
    model.fit(
        x_train,
        y_train,
        eval_set=eval_set,
        eval_metric="auc",
        callbacks=callbacks if callbacks else None,
    )

    result: dict[str, float | int] = {"best_iteration": int(getattr(model, "best_iteration_", -1) or -1)}
    if len(x_val) > 0:
        p_val = model.predict_proba(x_val)[:, 1]
        val_metrics = safe_binary_metrics(y_val, p_val)
        result["val_auc"] = float(val_metrics["auc"])
        result["val_logloss"] = float(val_metrics["logloss"])
    else:
        result["val_auc"] = float("nan")
        result["val_logloss"] = float("nan")
    return model, result


def _run_hyperopt(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_val: pd.DataFrame,
    y_val: pd.Series,
    args: argparse.Namespace,
) -> tuple[lgb.LGBMClassifier, dict[str, Any], pd.DataFrame]:
    if len(x_val) == 0:
        raise ValueError("Hyperopt requires a non-empty validation split.")

    rng = np.random.default_rng(int(args.random_state))
    trials: list[dict[str, Any]] = []
    best_model: lgb.LGBMClassifier | None = None
    best_params: dict[str, Any] | None = None
    best_auc = -np.inf
    best_trial = -1

    base_params = _build_base_lgbm_params(args)

    for trial in range(1, max(1, int(args.hyperopt_trials)) + 1):
        sampled = _sample_hyperopt_params(rng, args)
        params = {**base_params, **sampled}
        model, fit_info = _fit_lgbm_model(x_train, y_train, x_val, y_val, params, args)
        val_auc = float(fit_info["val_auc"])
        trial_row = {
            "trial": trial,
            "val_auc": val_auc,
            "val_logloss": float(fit_info["val_logloss"]),
            "best_iteration": int(fit_info["best_iteration"]),
            **sampled,
        }
        trials.append(trial_row)
        print(
            f"[hyperopt] trial={trial} val_auc={val_auc:.6f} "
            f"num_leaves={params['num_leaves']} lr={params['learning_rate']:.5f}"
        )
        if np.isfinite(val_auc) and val_auc > best_auc:
            best_auc = val_auc
            best_model = model
            best_params = params
            best_trial = trial

    if best_model is None or best_params is None:
        raise ValueError("Hyperopt failed to produce a valid best model.")

    tuning_df = pd.DataFrame(trials).sort_values("val_auc", ascending=False, ignore_index=True)
    summary = {
        "enabled": True,
        "trials": int(len(tuning_df)),
        "best_trial": int(best_trial),
        "best_val_auc": float(best_auc),
        "best_params": best_params,
    }
    return best_model, summary, tuning_df


def train_lgbm_and_rank(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame | None]:
    features = tp.INT_COLS + tp.CAT_COLS

    x_train = train_df[features]
    y_train = train_df[tp.LABEL_COL]
    x_val = val_df[features]
    y_val = val_df[tp.LABEL_COL]
    x_test = test_df[features]
    y_test = test_df[tp.LABEL_COL]

    tuning_results_df: pd.DataFrame | None = None
    if bool(args.enable_hyperopt):
        print(f"[hyperopt] running random search trials={args.hyperopt_trials}")
        model, hyperopt_summary, tuning_results_df = _run_hyperopt(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            args=args,
        )
    else:
        params = _build_base_lgbm_params(args)
        model, _ = _fit_lgbm_model(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            params=params,
            args=args,
        )
        hyperopt_summary = {
            "enabled": False,
            "trials": 0,
            "best_trial": None,
            "best_val_auc": None,
            "best_params": params,
        }

    pred_train = model.predict_proba(x_train)[:, 1]
    pred_val = model.predict_proba(x_val)[:, 1] if len(x_val) > 0 else np.array([])
    pred_test = model.predict_proba(x_test)[:, 1]

    metrics: dict[str, Any] = {
        "mode": "in_memory",
        "rows": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "encoding": args.cat_encoding,
        "hyperopt": hyperopt_summary,
        "train_metrics": safe_binary_metrics(y_train, pred_train),
        "test_metrics": safe_binary_metrics(y_test, pred_test),
    }
    if len(x_val) > 0:
        metrics["val_metrics"] = safe_binary_metrics(y_val, pred_val)

    booster = model.booster_
    gain = booster.feature_importance(importance_type="gain")
    split = booster.feature_importance(importance_type="split")
    names = booster.feature_name()

    fi = pd.DataFrame(
        {
            "feature": names,
            "gain": gain.astype(float),
            "split": split.astype(float),
        }
    ).sort_values("gain", ascending=False, ignore_index=True)
    fi["rank"] = np.arange(1, len(fi) + 1)
    total_gain = fi["gain"].sum()
    fi["gain_pct"] = np.where(total_gain > 0, (fi["gain"] / total_gain) * 100.0, 0.0)
    fi["from_hyperopt_best"] = bool(args.enable_hyperopt)

    return fi, metrics, tuning_results_df


def build_encoding_state_chunked(args: argparse.Namespace) -> tuple[EncodingState, dict[str, int]]:
    print("[chunked] pass1: building encoding state from train split")
    validate_split_sizes(args.train_size, args.val_size, args.test_size)

    rows_seen = 0
    rows_train = 0

    if args.cat_encoding == "frequency":
        counts: dict[str, Counter[str]] = {c: Counter() for c in tp.CAT_COLS}

        for i, (chunk, loaded_raw) in enumerate(
            iter_source_chunks(
                dsn=args.pg_dsn,
                source_table=args.source_table,
                source_partition_like=args.source_partition_like,
                max_rows=args.max_rows,
                fetch_size=args.fetch_size,
                cursor_name="eda_src_chunk_pass1",
            ),
            start=1,
        ):
            train_mask, _, _ = deterministic_split_masks(
                chunk, args.train_size, args.val_size, args.random_state
            )
            train_chunk = chunk.loc[train_mask]
            rows_seen += len(chunk)
            rows_train += len(train_chunk)

            if not train_chunk.empty:
                for c in tp.CAT_COLS:
                    vc = (
                        train_chunk[c]
                        .fillna(tp.MISSING_TOKEN)
                        .astype(str)
                        .value_counts(dropna=False)
                    )
                    counts[c].update(vc.to_dict())

            if i % max(1, args.log_every_chunks) == 0:
                print(
                    f"[chunked][pass1] chunks={i} rows_seen={rows_seen} "
                    f"train_rows={rows_train} raw_rows_seen={loaded_raw}"
                )

        if rows_train == 0:
            raise ValueError("No training rows found in pass1.")

        freq_maps = {
            c: {k: (v / rows_train) for k, v in counts[c].items()} for c in tp.CAT_COLS
        }
        state = EncodingState(
            freq_maps=freq_maps,
            counts=counts,
            train_rows=rows_train,
        )

    else:
        counts: dict[str, Counter[str]] = {c: Counter() for c in tp.CAT_COLS}
        int_maps: dict[str, dict[str, int]] = {c: {} for c in tp.CAT_COLS}
        next_ids: dict[str, int] = {c: 1 for c in tp.CAT_COLS}

        for i, (chunk, loaded_raw) in enumerate(
            iter_source_chunks(
                dsn=args.pg_dsn,
                source_table=args.source_table,
                source_partition_like=args.source_partition_like,
                max_rows=args.max_rows,
                fetch_size=args.fetch_size,
                cursor_name="eda_src_chunk_pass1",
            ),
            start=1,
        ):
            train_mask, _, _ = deterministic_split_masks(
                chunk, args.train_size, args.val_size, args.random_state
            )
            train_chunk = chunk.loc[train_mask]
            rows_seen += len(chunk)
            rows_train += len(train_chunk)

            if not train_chunk.empty:
                for c in tp.CAT_COLS:
                    vals = train_chunk[c].fillna(tp.MISSING_TOKEN).astype(str)
                    uniques = pd.unique(vals)
                    vc = vals.value_counts(dropna=False)
                    counts[c].update(vc.to_dict())
                    cmap = int_maps[c]
                    nid = next_ids[c]
                    for v in uniques:
                        if v not in cmap:
                            cmap[v] = nid
                            nid += 1
                    next_ids[c] = nid

            if i % max(1, args.log_every_chunks) == 0:
                mapped = sum(len(m) for m in int_maps.values())
                print(
                    f"[chunked][pass1] chunks={i} rows_seen={rows_seen} "
                    f"train_rows={rows_train} mapped_categories={mapped} "
                    f"raw_rows_seen={loaded_raw}"
                )

        if rows_train == 0:
            raise ValueError("No training rows found in pass1.")
        state = EncodingState(
            int_maps=int_maps,
            counts=counts,
            train_rows=rows_train,
        )

    split_info = {"rows_seen": rows_seen, "train_rows": rows_train}
    return state, split_info


def encode_features_chunk(
    frame: pd.DataFrame,
    state: EncodingState,
    cat_encoding: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    x = pd.DataFrame(index=frame.index)

    for c in tp.INT_COLS:
        x[c] = pd.to_numeric(frame[c], errors="coerce").fillna(0.0).astype(np.float32)

    if cat_encoding == "frequency":
        if state.freq_maps is None:
            raise ValueError("Frequency maps are missing in encoding state.")
        for c in tp.CAT_COLS:
            x[c] = (
                frame[c]
                .fillna(tp.MISSING_TOKEN)
                .astype(str)
                .map(state.freq_maps[c])
                .fillna(0.0)
                .astype(np.float32)
            )
    else:
        if state.int_maps is None:
            raise ValueError("Integer maps are missing in encoding state.")
        for c in tp.CAT_COLS:
            x[c] = (
                frame[c]
                .fillna(tp.MISSING_TOKEN)
                .astype(str)
                .map(state.int_maps[c])
                .fillna(0)
                .astype(np.int32)
            )

    y = frame[tp.LABEL_COL].astype(np.int8).to_numpy()
    return x, y


def _concat_eval_buffers(buffers: list[np.ndarray]) -> np.ndarray:
    if not buffers:
        return np.array([], dtype=np.float64)
    return np.concatenate(buffers, axis=0)


def train_lgbm_chunked_and_rank(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    state, pass1_info = build_encoding_state_chunked(args)

    print("[chunked] pass2: incremental LightGBM training")
    features = tp.INT_COLS + tp.CAT_COLS
    params = {
        "objective": "binary",
        "metric": ["auc", "binary_logloss"],
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": args.random_state,
        "num_threads": args.n_jobs,
        "verbosity": -1,
    }

    booster: lgb.Booster | None = None
    rows = {"train": 0, "val": 0, "test": 0}
    split_unknowns: dict[str, dict[str, Any]] = {
        "train": {
            "rows": 0,
            "unknown_by_feature": {c: 0 for c in tp.CAT_COLS},
        },
        "val": {
            "rows": 0,
            "unknown_by_feature": {c: 0 for c in tp.CAT_COLS},
        },
        "test": {
            "rows": 0,
            "unknown_by_feature": {c: 0 for c in tp.CAT_COLS},
        },
    }

    val_x_buffers: list[np.ndarray] = []
    val_y_buffers: list[np.ndarray] = []
    test_x_buffers: list[np.ndarray] = []
    test_y_buffers: list[np.ndarray] = []

    val_kept = 0
    test_kept = 0

    for i, (chunk, loaded_raw) in enumerate(
        iter_source_chunks(
            dsn=args.pg_dsn,
            source_table=args.source_table,
            source_partition_like=args.source_partition_like,
            max_rows=args.max_rows,
            fetch_size=args.fetch_size,
            cursor_name="eda_src_chunk_pass2",
        ),
        start=1,
    ):
        train_mask, val_mask, test_mask = deterministic_split_masks(
            chunk, args.train_size, args.val_size, args.random_state
        )

        train_chunk = chunk.loc[train_mask]
        val_chunk = chunk.loc[val_mask]
        test_chunk = chunk.loc[test_mask]

        rows["train"] += len(train_chunk)
        rows["val"] += len(val_chunk)
        rows["test"] += len(test_chunk)
        split_unknowns["train"]["rows"] += len(train_chunk)
        split_unknowns["val"]["rows"] += len(val_chunk)
        split_unknowns["test"]["rows"] += len(test_chunk)

        train_unknown = count_unknown_categories(train_chunk, state, args.cat_encoding)
        val_unknown = count_unknown_categories(val_chunk, state, args.cat_encoding)
        test_unknown = count_unknown_categories(test_chunk, state, args.cat_encoding)
        for c in tp.CAT_COLS:
            split_unknowns["train"]["unknown_by_feature"][c] += train_unknown[c]
            split_unknowns["val"]["unknown_by_feature"][c] += val_unknown[c]
            split_unknowns["test"]["unknown_by_feature"][c] += test_unknown[c]

        if not train_chunk.empty:
            x_train, y_train = encode_features_chunk(train_chunk, state, args.cat_encoding)
            dtrain = lgb.Dataset(
                x_train,
                label=y_train,
                feature_name=features,
                free_raw_data=True,
            )
            booster = lgb.train(
                params=params,
                train_set=dtrain,
                num_boost_round=args.chunk_boost_rounds,
                init_model=booster,
                keep_training_booster=True,
            )

        if val_kept < args.eval_max_rows and not val_chunk.empty:
            take = min(args.eval_max_rows - val_kept, len(val_chunk))
            x_val, y_val = encode_features_chunk(
                val_chunk.iloc[:take], state, args.cat_encoding
            )
            val_x_buffers.append(x_val.to_numpy(dtype=np.float32, copy=False))
            val_y_buffers.append(y_val)
            val_kept += take

        if test_kept < args.eval_max_rows and not test_chunk.empty:
            take = min(args.eval_max_rows - test_kept, len(test_chunk))
            x_test, y_test = encode_features_chunk(
                test_chunk.iloc[:take], state, args.cat_encoding
            )
            test_x_buffers.append(x_test.to_numpy(dtype=np.float32, copy=False))
            test_y_buffers.append(y_test)
            test_kept += take

        if i % max(1, args.log_every_chunks) == 0:
            print(
                f"[chunked][pass2] chunks={i} train_rows={rows['train']} "
                f"val_rows={rows['val']} test_rows={rows['test']} "
                f"val_eval_kept={val_kept} test_eval_kept={test_kept} "
                f"raw_rows_seen={loaded_raw}"
            )

    if booster is None:
        raise ValueError("Training produced no booster. Check filters/splits.")

    fi = pd.DataFrame(
        {
            "feature": booster.feature_name(),
            "gain": booster.feature_importance(importance_type="gain").astype(float),
            "split": booster.feature_importance(importance_type="split").astype(float),
        }
    ).sort_values("gain", ascending=False, ignore_index=True)
    fi["rank"] = np.arange(1, len(fi) + 1)
    total_gain = fi["gain"].sum()
    fi["gain_pct"] = np.where(total_gain > 0, (fi["gain"] / total_gain) * 100.0, 0.0)

    metrics: dict[str, Any] = {
        "mode": "chunked",
        "encoding": args.cat_encoding,
        "rows": rows,
        "eval_rows_kept": {"val": int(val_kept), "test": int(test_kept)},
        "chunk_boost_rounds": args.chunk_boost_rounds,
        "pass1": pass1_info,
    }

    for split_name, split_stats in split_unknowns.items():
        total_unknown = int(sum(split_stats["unknown_by_feature"].values()))
        rows_seen = int(split_stats["rows"])
        denom = max(rows_seen * len(tp.CAT_COLS), 1)
        split_stats["unknown_total"] = total_unknown
        split_stats["unknown_rate"] = float(total_unknown / denom)

    encoding_report = build_encoding_report(
        state=state,
        cat_encoding=args.cat_encoding,
        top_k=args.encoding_debug_top_k,
        unknown_stats=split_unknowns,
    )
    metrics["encoding_report"] = encoding_report

    if val_x_buffers:
        x_val_eval = _concat_eval_buffers(val_x_buffers)
        y_val_eval = _concat_eval_buffers(val_y_buffers)
        p_val = booster.predict(x_val_eval)
        metrics["val_metrics"] = safe_binary_metrics(y_val_eval, p_val)

    if test_x_buffers:
        x_test_eval = _concat_eval_buffers(test_x_buffers)
        y_test_eval = _concat_eval_buffers(test_y_buffers)
        p_test = booster.predict(x_test_eval)
        metrics["test_metrics"] = safe_binary_metrics(y_test_eval, p_test)

    if bool(args.print_encoding_summary):
        print_encoding_summary(encoding_report)

    return fi, metrics


def save_outputs(
    fi: pd.DataFrame,
    metrics: dict[str, Any],
    args: argparse.Namespace,
    tuning_results_df: pd.DataFrame | None = None,
) -> None:
    out_csv = Path(args.importance_csv)
    out_json = Path(args.metrics_json)
    out_encoding = Path(args.encoding_report_json)
    out_tuning = Path(args.tuning_results_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_encoding.parent.mkdir(parents=True, exist_ok=True)
    out_tuning.parent.mkdir(parents=True, exist_ok=True)

    fi.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    encoding_report = metrics.get("encoding_report", {})
    out_encoding.write_text(json.dumps(encoding_report, indent=2), encoding="utf-8")
    if tuning_results_df is not None and not tuning_results_df.empty:
        tuning_results_df.to_csv(out_tuning, index=False)
        print(f"[saved] tuning_results_csv={out_tuning}")

    print("[result] top features by gain")
    print(fi.head(args.top_n).to_string(index=False))
    print(f"[saved] importance_csv={out_csv}")
    print(f"[saved] metrics_json={out_json}")
    print(f"[saved] encoding_report_json={out_encoding}")


def main() -> None:
    args = parse_args()
    validate_split_sizes(args.train_size, args.val_size, args.test_size)
    tuning_results_df: pd.DataFrame | None = None

    if args.train_mode == "in_memory":
        print("[mode] in_memory")
        print("[step] loading source data")
        df = load_source_df(
            dsn=args.pg_dsn,
            source_table=args.source_table,
            source_partition_like=args.source_partition_like,
            max_rows=args.max_rows,
            fetch_size=args.fetch_size,
        )
        print(f"[data] rows={len(df)}")

        print("[step] splitting train/val/test")
        train_df, val_df, test_df = split_3way(
            df=df,
            train_size=args.train_size,
            val_size=args.val_size,
            test_size=args.test_size,
            random_state=args.random_state,
        )
        print(f"[split] train={len(train_df)} val={len(val_df)} test={len(test_df)}")

        print(f"[step] encoding categorical columns: {args.cat_encoding}")
        state = fit_encoding_state(train_df, args.cat_encoding)
        split_unknowns: dict[str, dict[str, Any]] = {
            "train": {
                "rows": int(len(train_df)),
                "unknown_by_feature": count_unknown_categories(
                    train_df, state, args.cat_encoding
                ),
            },
            "val": {
                "rows": int(len(val_df)),
                "unknown_by_feature": count_unknown_categories(
                    val_df, state, args.cat_encoding
                ),
            },
            "test": {
                "rows": int(len(test_df)),
                "unknown_by_feature": count_unknown_categories(
                    test_df, state, args.cat_encoding
                ),
            },
        }
        for split_name, split_stats in split_unknowns.items():
            total_unknown = int(sum(split_stats["unknown_by_feature"].values()))
            rows_seen = int(split_stats["rows"])
            denom = max(rows_seen * len(tp.CAT_COLS), 1)
            split_stats["unknown_total"] = total_unknown
            split_stats["unknown_rate"] = float(total_unknown / denom)

        train_df = encode_frame_with_state(train_df, state, args.cat_encoding)
        val_df = encode_frame_with_state(val_df, state, args.cat_encoding)
        test_df = encode_frame_with_state(test_df, state, args.cat_encoding)

        for frame in (train_df, val_df, test_df):
            ensure_numeric_features(frame)

        print("[step] training LightGBM and computing gain importance")
        fi, metrics, tuning_results_df = train_lgbm_and_rank(train_df, val_df, test_df, args)
        encoding_report = build_encoding_report(
            state=state,
            cat_encoding=args.cat_encoding,
            top_k=args.encoding_debug_top_k,
            unknown_stats=split_unknowns,
        )
        metrics["encoding_report"] = encoding_report
        if bool(args.print_encoding_summary):
            print_encoding_summary(encoding_report)
    else:
        print("[mode] chunked")
        print("[step] streaming full data with bounded memory")
        fi, metrics = train_lgbm_chunked_and_rank(args)
        if bool(args.enable_hyperopt):
            print("[hyperopt] ignored in chunked mode; use --train-mode in_memory.")
            metrics["hyperopt"] = {
                "enabled": False,
                "ignored": True,
                "reason": "chunked_mode_not_supported",
            }

    save_outputs(fi, metrics, args, tuning_results_df=tuning_results_df)


if __name__ == "__main__":
    main()
