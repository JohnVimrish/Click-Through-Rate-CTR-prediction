from __future__ import annotations

import argparse
import io
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl
import psycopg
from psycopg import sql

import uuid
from typing import Iterator

LABEL_COL = "label"
INT_COLS = [f"int_feature_{i}" for i in range(1, 14)]
CAT_COLS = [f"cat_feature_{i}" for i in range(1, 27)]
BASE_COLS = [LABEL_COL] + INT_COLS + CAT_COLS

MISSING_TOKEN = "__MISSING__"
RARE_TOKEN = "__RARE__"

@dataclass(frozen=True)
class DbTable:
    schema: str
    table: str

def fetch_source_partitions(
    conn: psycopg.Connection,
    source: DbTable,
    source_partition_like: str,
) -> list[str]:
    q = sql.SQL(
        "SELECT DISTINCT partition_file FROM {} WHERE partition_file LIKE %s ORDER BY 1"
    ).format(sql.Identifier(source.schema, source.table))
    with conn.cursor() as cur:
        cur.execute(q, (source_partition_like,))
        return [r[0] for r in cur.fetchall() if r[0] is not None]


def read_partition_batches(
    conn: psycopg.Connection,
    source: DbTable,
    source_partition: str,
    batch_rows: int,
) -> Iterator[pl.DataFrame]:
    cols = [LABEL_COL] + INT_COLS + CAT_COLS
    q = sql.SQL("SELECT {} FROM {} WHERE partition_file = %s").format(
        sql.SQL(", ").join(map(sql.Identifier, cols)),
        sql.Identifier(source.schema, source.table),
    )
    # Keep the server cursor across commits performed by COPY batches.
    with conn.cursor(name=f"src_{uuid.uuid4().hex[:10]}", withhold=True) as cur:
        cur.execute(q, (source_partition,))
        while True:
            rows = cur.fetchmany(batch_rows)
            if not rows:
                break
            yield pl.DataFrame(rows, schema=cols, orient="row")

def split_qualified(name: str) -> DbTable:
    parts = name.split(".", 1)
    if len(parts) == 2:
        return DbTable(parts[0], parts[1])
    return DbTable("public", parts[0])

def sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    chunk = stem.split("_")
    if len(chunk) > 1 and chunk[-1].isdigit():
        return (int(chunk[-1]), stem)
    return (10**9, stem)

def list_input_files(dataset_dir: Path, pattern: str) -> list[Path]:
    files = sorted(dataset_dir.glob(pattern), key=sort_key)
    return [f for f in files if f.is_file()]

def scan_file(path: Path, n_rows: int | None = None) -> pl.LazyFrame:
    kwargs: dict[str, Any] = {
        "source": path,
        "has_header": False,
        "separator": "\t",
        "new_columns": BASE_COLS,
        "infer_schema_length": 200,
        "truncate_ragged_lines": True,
        "ignore_errors": True,
    }
    if n_rows is not None and n_rows > 0:
        kwargs["n_rows"] = n_rows
    return pl.scan_csv(**kwargs)

def normalize_lazy(lf: pl.LazyFrame) -> pl.LazyFrame:
    cat_exprs = [
        pl.col(c).cast(pl.Utf8, strict=False).str.strip_chars().replace("", None).alias(c)
        for c in CAT_COLS
    ]
    return lf.with_columns(
        [pl.col(LABEL_COL).cast(pl.Int16, strict=False)]
        + [pl.col(c).cast(pl.Int64, strict=False).alias(c) for c in INT_COLS]
        + cat_exprs
    )

def build_train_lazy(train_files: list[Path], fit_rows_per_file: int | None) -> pl.LazyFrame:
    lfs = [normalize_lazy(scan_file(p, fit_rows_per_file)) for p in train_files]
    if not lfs:
        raise ValueError("No training files were provided.")
    return pl.concat(lfs, how="vertical_relaxed")

def fit_int_medians(train_lf: pl.LazyFrame) -> dict[str, int]:
    med_exprs = [pl.col(c).median().alias(c) for c in INT_COLS]
    med_df = train_lf.select(med_exprs).collect(streaming=True)
    medians: dict[str, int] = {}
    for c in INT_COLS:
        v = med_df[c][0]
        medians[c] = int(v) if v is not None else 0
    return medians

def fit_cat_levels(
    train_lf: pl.LazyFrame,
    min_freq: int,
    max_levels: int,
) -> dict[str, list[str]]:
    levels: dict[str, list[str]] = {}
    for c in CAT_COLS:
        count_df = (
            train_lf.select(pl.col(c).fill_null(MISSING_TOKEN).alias(c))
            .group_by(c)
            .len()
            .sort("len", descending=True)
            .collect(streaming=True)
        )
        kept = count_df.filter(pl.col("len") >= min_freq)
        if max_levels > 0:
            kept = kept.head(max_levels)
        vals = [str(v) for v in kept[c].to_list() if v is not None]
        if MISSING_TOKEN not in vals:
            vals.append(MISSING_TOKEN)
        levels[c] = vals
    return levels

def save_artifacts(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

def load_artifacts(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Artifact file not found: {path}")
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        raise ValueError(
            f"Artifact file is empty: {path}. Generate artifacts first or set --artifact-path to a valid JSON artifact."
        )
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid artifact JSON at {path}: {exc}") from exc

def fit_artifacts(
    train_files: list[Path],
    artifact_path: Path,
    min_cat_freq: int,
    max_cat_levels: int,
    fit_rows_per_file: int | None,
) -> dict[str, Any]:
    train_lf = build_train_lazy(train_files, fit_rows_per_file)
    int_medians = fit_int_medians(train_lf)
    cat_levels = fit_cat_levels(train_lf, min_cat_freq, max_cat_levels)

    payload: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "fit_rows_per_file": fit_rows_per_file,
        "min_cat_freq": min_cat_freq,
        "max_cat_levels": max_cat_levels,
        "int_medians": int_medians,
        "cat_levels": cat_levels,
    }
    save_artifacts(artifact_path, payload)
    return payload

def batch_reader(path: Path, batch_rows: int) -> pl.io.csv.batched_reader.BatchedCsvReader:
    return pl.read_csv_batched(
        source=path,
        has_header=False,
        separator="\t",
        new_columns=BASE_COLS,
        infer_schema_length=200,
        truncate_ragged_lines=True,
        ignore_errors=True,
        batch_size=batch_rows,
    )

def normalize_batch(df: pl.DataFrame) -> pl.DataFrame:
    cat_exprs = [
        pl.col(c).cast(pl.Utf8, strict=False).str.strip_chars().replace("", None).alias(c)
        for c in CAT_COLS
    ]
    return df.with_columns(
        [pl.col(LABEL_COL).cast(pl.Int16, strict=False)]
        + [pl.col(c).cast(pl.Int64, strict=False).alias(c) for c in INT_COLS]
        + cat_exprs
    )

def transform_batch(
    df: pl.DataFrame,
    file_name: str,
    artifacts: dict[str, Any],
    add_missing_flags: bool,
) -> pl.DataFrame:
    df = normalize_batch(df)
    int_medians: dict[str, int] = artifacts["int_medians"]
    cat_levels: dict[str, list[str]] = artifacts["cat_levels"]

    int_fill_exprs = [pl.col(c).fill_null(int_medians.get(c, 0)).alias(c) for c in INT_COLS]
    int_flag_exprs = []
    if add_missing_flags:
        int_flag_exprs = [
            pl.col(c).is_null().cast(pl.Int8).alias(f"{c}_was_null") for c in INT_COLS
        ]

    cat_exprs = []
    for c in CAT_COLS:
        keep_vals = cat_levels.get(c, [MISSING_TOKEN])
        val_expr = pl.col(c).fill_null(MISSING_TOKEN)
        cat_exprs.append(
            pl.when(val_expr.is_in(keep_vals))
            .then(val_expr)
            .otherwise(pl.lit(RARE_TOKEN))
            .alias(c)
        )

    df = df.with_columns(int_fill_exprs + int_flag_exprs + cat_exprs)
    df = df.with_columns(pl.lit(file_name).alias("partition_file"))

    out_cols = (
        [LABEL_COL]
        + INT_COLS
        + ([f"{c}_was_null" for c in INT_COLS] if add_missing_flags else [])
        + CAT_COLS
        + ["partition_file"]
    )
    return df.select(out_cols)

def ensure_schema(conn: psycopg.Connection, schema_name: str) -> None:
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(schema_name))
        )

def ensure_tables(
    conn: psycopg.Connection,
    target: DbTable,
    control: DbTable,
    add_missing_flags: bool,
) -> None:
    ensure_schema(conn, target.schema)
    ensure_schema(conn, control.schema)

    int_cols_sql = ",\n".join([f"{c} BIGINT NULL" for c in INT_COLS])
    cat_cols_sql = ",\n".join([f"{c} TEXT NULL" for c in CAT_COLS])
    flag_cols_sql = ",\n".join(
        [f"{c}_was_null SMALLINT NOT NULL DEFAULT 0" for c in INT_COLS]
    )

    extras = []
    if add_missing_flags:
        extras.append(flag_cols_sql)
    extras.append(cat_cols_sql)
    features_sql = ",\n".join(extras)

    target_ident = sql.Identifier(target.schema, target.table)
    control_ident = sql.Identifier(control.schema, control.table)
    idx_ident = sql.Identifier(f"idx_{target.table}_partition_file")

    create_target = sql.SQL(
        """
        CREATE TABLE IF NOT EXISTS {} (
            label SMALLINT NULL,
            {},
            {},
            partition_file TEXT NOT NULL,
            loaded_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    ).format(target_ident, sql.SQL(int_cols_sql), sql.SQL(features_sql))

    create_control = sql.SQL(
        """
        CREATE TABLE IF NOT EXISTS {} (
            file_name TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            rows_loaded BIGINT NOT NULL DEFAULT 0,
            started_at TIMESTAMPTZ,
            finished_at TIMESTAMPTZ,
            error_message TEXT
        )
        """
    ).format(control_ident)

    create_idx = sql.SQL(
        """
        CREATE INDEX IF NOT EXISTS {}
        ON {} (partition_file)
        """
    ).format(idx_ident, target_ident)

    with conn.cursor() as cur:
        cur.execute(create_target)
        cur.execute(create_control)
        cur.execute(create_idx)
    conn.commit()

def mark_processing(conn: psycopg.Connection, control: DbTable, file_name: str) -> None:
    q = sql.SQL(
        """
        INSERT INTO {}(file_name, status, rows_loaded, started_at, finished_at, error_message)
        VALUES (%s, 'processing', 0, now(), NULL, NULL)
        ON CONFLICT (file_name) DO UPDATE
        SET status = 'processing',
            rows_loaded = 0,
            started_at = now(),
            finished_at = NULL,
            error_message = NULL
        """
    ).format(sql.Identifier(control.schema, control.table))
    with conn.cursor() as cur:
        cur.execute(q, (file_name,))
    conn.commit()



def mark_processed(
    conn: psycopg.Connection,
    control: DbTable,
    file_name: str,
    rows_loaded: int,
) -> None:
    q = sql.SQL(
        """
        UPDATE {}
        SET status = 'processed',
            rows_loaded = %s,
            finished_at = now(),
            error_message = NULL
        WHERE file_name = %s
        """
    ).format(sql.Identifier(control.schema, control.table))
    with conn.cursor() as cur:
        cur.execute(q, (rows_loaded, file_name))
    conn.commit()




def mark_failed(conn: psycopg.Connection, control: DbTable, file_name: str, err: str) -> None:
    q = sql.SQL(
        """
        UPDATE {}
        SET status = 'failed',
            finished_at = now(),
            error_message = %s
        WHERE file_name = %s
        """
    ).format(sql.Identifier(control.schema, control.table))
    with conn.cursor() as cur:
        cur.execute(q, (err[:4000], file_name))
    conn.commit()



def get_processed_files(conn: psycopg.Connection, control: DbTable) -> set[str]:
    q = sql.SQL("SELECT file_name FROM {} WHERE status = 'processed'").format(
        sql.Identifier(control.schema, control.table)
    )
    with conn.cursor() as cur:
        cur.execute(q)
        return {r[0] for r in cur.fetchall()}



def copy_batch(
    conn: psycopg.Connection,
    target: DbTable,
    df: pl.DataFrame,
    add_missing_flags: bool,
) -> int:
    if df.height == 0:
        return 0

    copy_cols = (
        [LABEL_COL]
        + INT_COLS
        + ([f"{c}_was_null" for c in INT_COLS] if add_missing_flags else [])
        + CAT_COLS
        + ["partition_file"]
    )

    buffer = io.StringIO()
    df.write_csv(
        file=buffer,
        include_header=False,
        separator="\t",
        null_value="\\N",
    )
    buffer.seek(0)

    copy_stmt = sql.SQL(
        "COPY {} ({}) FROM STDIN WITH (FORMAT csv, DELIMITER E'\\t', NULL '\\N')"
    ).format(
        sql.Identifier(target.schema, target.table),
        sql.SQL(", ").join(map(sql.Identifier, copy_cols)),
    )

    with conn.cursor() as cur:
        with cur.copy(copy_stmt) as cp:
            cp.write(buffer.read())
    conn.commit()
    return df.height



def process_file(
    conn: psycopg.Connection,
    target: DbTable,
    control: DbTable,
    path: Path,
    artifacts: dict[str, Any],
    batch_rows: int,
    add_missing_flags: bool,
) -> None:
    file_name = path.name
    mark_processing(conn, control, file_name)
    total_rows = 0
    try:
        reader = batch_reader(path, batch_rows)
        while True:
            batches = reader.next_batches(1)
            if not batches:
                break
            batch = batches[0]
            if batch.height == 0:
                continue
            transformed = transform_batch(batch, file_name, artifacts, add_missing_flags)
            total_rows += copy_batch(conn, target, transformed, add_missing_flags)

        mark_processed(conn, control, file_name, total_rows)
        print(f"[processed] {file_name} rows={total_rows}")
    except Exception as exc:
        conn.rollback()
        mark_failed(conn, control, file_name, str(exc))
        print(f"[failed] {file_name} error={exc}")



def run_transform(
    artifacts: dict[str, Any],
    dsn: str,
    source_table: str,
    target_table: str,
    control_table: str,
    batch_rows: int,
    partition_suffix: str,
    source_partition_like: str,
    add_missing_flags: bool,
) -> None:
    source = split_qualified(source_table)
    target = split_qualified(target_table)
    control = split_qualified(control_table)

    with psycopg.connect(dsn) as conn:
        ensure_tables(conn, target, control, add_missing_flags)

        all_parts = fetch_source_partitions(conn, source, source_partition_like)
        processed = get_processed_files(conn, control)
        pending = [p for p in all_parts if p not in processed]

        print(f"source_partitions={len(all_parts)} pending={len(pending)}")
        for part in pending:
            mark_processing(conn, control, part)
            total_rows = 0
            try:
                for batch in read_partition_batches(conn, source, part, batch_rows):
                    transformed = transform_batch(
                        batch,
                        f"{part}{partition_suffix}",  # <- suffix _2 here
                        artifacts,
                        add_missing_flags,
                    )
                    total_rows += copy_batch(conn, target, transformed, add_missing_flags)

                mark_processed(conn, control, part, total_rows)
                print(f"[processed] {part} rows={total_rows}")
            except Exception as exc:
                conn.rollback()
                mark_failed(conn, control, part, str(exc))
                print(f"[failed] {part} error={exc}")

def fit_int_medians_from_source(
    conn: psycopg.Connection,
    source: DbTable,
    source_partition_like: str,
) -> dict[str, int]:
    medians: dict[str, int] = {}
    source_ident = sql.Identifier(source.schema, source.table)
    with conn.cursor() as cur:
        for c in INT_COLS:
            q = sql.SQL(
                "SELECT percentile_cont(0.5) WITHIN GROUP (ORDER BY {}) FROM {} WHERE partition_file LIKE %s"
            ).format(sql.Identifier(c), source_ident)
            cur.execute(q, (source_partition_like,))
            v = cur.fetchone()[0]
            medians[c] = int(v) if v is not None else 0
    return medians

def fit_cat_levels_from_source(
    conn: psycopg.Connection,
    source: DbTable,
    min_cat_freq: int,
    max_cat_levels: int,
    source_partition_like: str,
) -> dict[str, list[str]]:
    levels: dict[str, list[str]] = {}
    source_ident = sql.Identifier(source.schema, source.table)
    with conn.cursor() as cur:
        for c in CAT_COLS:
            q = sql.SQL(
                """
                SELECT COALESCE(CAST({} AS text), %s) AS value, COUNT(*) AS cnt
                FROM {}
                WHERE partition_file LIKE %s
                GROUP BY 1
                HAVING COUNT(*) >= %s
                ORDER BY cnt DESC
                LIMIT %s
                """
            ).format(sql.Identifier(c), source_ident)
            cur.execute(
                q,
                (MISSING_TOKEN, source_partition_like, min_cat_freq, max_cat_levels),
            )
            vals = [r[0] for r in cur.fetchall() if r[0] is not None]
            if MISSING_TOKEN not in vals:
                vals.append(MISSING_TOKEN)
            levels[c] = vals
    return levels

def fit_artifacts_from_source(
    conn: psycopg.Connection,
    source: DbTable,
    min_cat_freq: int,
    max_cat_levels: int,
    source_partition_like: str,
) -> dict[str, Any]:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_table": f"{source.schema}.{source.table}",
        "min_cat_freq": min_cat_freq,
        "max_cat_levels": max_cat_levels,
        "source_partition_like": source_partition_like,
        "int_medians": fit_int_medians_from_source(conn, source, source_partition_like),
        "cat_levels": fit_cat_levels_from_source(
            conn,
            source,
            min_cat_freq,
            max_cat_levels,
            source_partition_like,
        ),
    }

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transform data from source table into analytics target with psycopg3 COPY.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--mode",
        choices=["transform"],
        default=os.getenv("MODE", "transform"),
    )

    parser.add_argument("--source-table", default=os.getenv("SOURCE_TABLE", "mlops.ml_features_full_part_test"))
    parser.add_argument("--target-table", default=os.getenv("TARGET_TABLE", "mlops.analytics_mlops"))
    parser.add_argument("--partition-suffix", default=os.getenv("PARTITION_SUFFIX", "_2"))
    parser.add_argument(
        "--source-partition-like",
        default=os.getenv("SOURCE_PARTITION_LIKE", "%"),
        help="SQL LIKE pattern to filter source partitions, e.g. 'day_5%' or '%_2'.",
    )

    parser.add_argument(
        "--artifact-path",
        default=os.getenv(
            "ARTIFACT_PATH",
            "/app/artifacts/preprocess_artifacts.json",
        ),
    )
    parser.add_argument("--pg-dsn", default=os.getenv("PG_DSN", "postgresql://local_user:Abc$12345@docker-lcpostgres:5432/migration_db_test"))

    parser.add_argument(
        "--control-table",
        default=os.getenv("CONTROL_TABLE", "mlops.ingest_file_status"),
    )
    parser.add_argument("--batch-rows", type=int, default=int(os.getenv("BATCH_ROWS", "200000")))
    parser.add_argument(
        "--min-cat-freq", type=int, default=int(os.getenv("MIN_CAT_FREQ", "50"))
    )
    parser.add_argument(
        "--max-cat-levels", type=int, default=int(os.getenv("MAX_CAT_LEVELS", "500"))
    )
    parser.add_argument(
        "--add-missing-flags",
        type=int,
        default=int(os.getenv("ADD_MISSING_FLAGS", "1")),
        choices=[0, 1],
    )
    args, _ = parser.parse_known_args()
    return args

def main() -> None:
    args = parse_args()
    artifact_path = Path(args.artifact_path)
    add_missing_flags = bool(args.add_missing_flags)

    if not args.pg_dsn:
        raise ValueError("PG_DSN is required.")

    try:
        artifacts = load_artifacts(artifact_path)
        print(f"[transform] artifact loaded from {artifact_path}")
    except (FileNotFoundError, ValueError) as exc:
        print(f"[fit] {exc}")
        print("[fit] building artifacts from source table")

        with psycopg.connect(args.pg_dsn) as fit_conn:
            artifacts = fit_artifacts_from_source(
                conn=fit_conn,
                source=split_qualified(args.source_table),
                min_cat_freq=args.min_cat_freq,
                max_cat_levels=args.max_cat_levels,
                source_partition_like=args.source_partition_like,
            )
        save_artifacts(artifact_path, artifacts)
        print(f"[fit] artifact saved to {artifact_path}")

    run_transform(
        artifacts=artifacts,
        dsn=args.pg_dsn,
        source_table=args.source_table,
        target_table=args.target_table,
        control_table=args.control_table,
        batch_rows=args.batch_rows,
        partition_suffix=args.partition_suffix,
        source_partition_like=args.source_partition_like,
        add_missing_flags=add_missing_flags,
    )




if __name__ == "__main__":
    main()
