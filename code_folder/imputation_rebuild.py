from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import polars as pl
import psycopg
from psycopg import sql
import transform_pipeline as tp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild analytics target with log1p-transformed integer features "
            "using the same source/filter flow as transform_pipeline."
        ),
        allow_abbrev=False,
    )
    parser.add_argument(
        "--source-table",
        default=os.getenv("SOURCE_TABLE", "mlops.ml_features_full_dump"),
    )
    parser.add_argument(
        "--target-table",
        default=os.getenv("TARGET_TABLE", "mlops.ml_features_full_part_test"),
    )
    parser.add_argument(
        "--control-table",
        default=os.getenv("CONTROL_TABLE", "mlops.transform_log1p_status"),
    )
    parser.add_argument(
        "--source-partition-like",
        default=os.getenv("SOURCE_PARTITION_LIKE", "day_2%"),
        help="SQL LIKE pattern to filter source partitions, e.g. 'day_5%%' or '%%_2'.",
    )
    parser.add_argument(
        "--partition-suffix",
        default=os.getenv("PARTITION_SUFFIX", "_2"),
    )
    parser.add_argument(
        "--artifact-path",
        default=os.getenv("ARTIFACT_PATH", "/app/artifacts/preprocess_artifacts.json"),
    )
    parser.add_argument(
        "--pg-dsn",
        default=os.getenv(
            "PG_DSN",
            "postgresql://local_user:Abc$12345@docker-lcpostgres:5432/migration_db_test",
        ),
    )
    parser.add_argument(
        "--batch-rows",
        type=int,
        default=int(os.getenv("BATCH_ROWS", "250000")),
    )
    parser.add_argument(
        "--min-cat-freq",
        type=int,
        default=int(os.getenv("MIN_CAT_FREQ", "50")),
    )
    parser.add_argument(
        "--max-cat-levels",
        type=int,
        default=int(os.getenv("MAX_CAT_LEVELS", "500")),
    )
    parser.add_argument(
        "--add-missing-flags",
        type=int,
        default=int(os.getenv("ADD_MISSING_FLAGS", "0")),
        choices=[0, 1],
    )
    parser.add_argument(
        "--truncate-target",
        type=int,
        default=int(os.getenv("TRUNCATE_TARGET", "1")),
        choices=[0, 1],
        help="Truncate target table before reload.",
    )
    parser.add_argument(
        "--drop-indexes",
        type=int,
        default=int(os.getenv("DROP_INDEXES", "1")),
        choices=[0, 1],
        help="Drop existing target indexes before reload for faster COPY.",
    )
    parser.add_argument(
        "--recreate-indexes",
        type=int,
        default=int(os.getenv("RECREATE_INDEXES", "1")),
        choices=[0, 1],
        help="Recreate dropped indexes after reload.",
    )
    parser.add_argument(
        "--reset-control-table",
        type=int,
        default=int(os.getenv("RESET_CONTROL_TABLE", "1")),
        choices=[0, 1],
        help="Truncate control table before reload so all source partitions are reprocessed.",
    )
    parser.add_argument(
        "--ensure-float-int-cols",
        type=int,
        default=int(os.getenv("ENSURE_FLOAT_INT_COLS", "1")),
        choices=[0, 1],
        help="Alter int_feature_* columns in target table to DOUBLE PRECISION for log1p values.",
    )
    return parser.parse_args()


def transform_batch_log1p(
    df: pl.DataFrame,
    file_name: str,
    artifacts: dict[str, Any],
    add_missing_flags: bool,
) -> pl.DataFrame:
    df = tp.normalize_batch(df)
    int_medians: dict[str, int] = artifacts["int_medians"]
    cat_levels: dict[str, list[str]] = artifacts["cat_levels"]

    int_exprs = []
    int_flag_exprs = []
    for c in tp.INT_COLS:
        base_expr = pl.col(c).fill_null(float(int_medians.get(c, 0))).cast(
            pl.Float64, strict=False
        )
        # Guard against unexpected negatives so log1p stays valid.
        int_exprs.append(
            pl.when(base_expr < 0.0)
            .then(pl.lit(0.0))
            .otherwise(base_expr)
            .log1p()
            .alias(c)
        )
        if add_missing_flags:
            int_flag_exprs.append(pl.col(c).is_null().cast(pl.Int8).alias(f"{c}_was_null"))

    cat_exprs = []
    for c in tp.CAT_COLS:
        keep_vals = cat_levels.get(c, [tp.MISSING_TOKEN])
        val_expr = pl.col(c).fill_null(tp.MISSING_TOKEN)
        cat_exprs.append(
            pl.when(val_expr.is_in(keep_vals))
            .then(val_expr)
            .otherwise(pl.lit(tp.RARE_TOKEN))
            .alias(c)
        )

    df = df.with_columns(int_exprs + int_flag_exprs + cat_exprs)
    df = df.with_columns(pl.lit(file_name).alias("partition_file"))

    out_cols = (
        [tp.LABEL_COL]
        + tp.INT_COLS
        + ([f"{c}_was_null" for c in tp.INT_COLS] if add_missing_flags else [])
        + tp.CAT_COLS
        + ["partition_file"]
    )
    return df.select(out_cols)


def snapshot_target_indexes(
    conn: psycopg.Connection,
    target: tp.DbTable,
) -> list[tuple[str, str]]:
    q = """
        SELECT indexname, indexdef
        FROM pg_indexes
        WHERE schemaname = %s AND tablename = %s
        ORDER BY indexname
    """
    with conn.cursor() as cur:
        cur.execute(q, (target.schema, target.table))
        return [(r[0], r[1]) for r in cur.fetchall()]


def drop_target_indexes(
    conn: psycopg.Connection,
    target: tp.DbTable,
    indexes: list[tuple[str, str]],
) -> None:
    if not indexes:
        print("[indexes] no indexes found")
        return
    with conn.cursor() as cur:
        for index_name, _ in indexes:
            cur.execute(
                sql.SQL("DROP INDEX IF EXISTS {}").format(
                    sql.Identifier(target.schema, index_name)
                )
            )
    conn.commit()
    print(f"[indexes] dropped={len(indexes)}")


def recreate_indexes(conn: psycopg.Connection, indexes: list[tuple[str, str]]) -> None:
    if not indexes:
        return
    with conn.cursor() as cur:
        for _, index_def in indexes:
            cur.execute(index_def)
    conn.commit()
    print(f"[indexes] recreated={len(indexes)}")


def truncate_table(conn: psycopg.Connection, table: tp.DbTable) -> None:
    with conn.cursor() as cur:
        cur.execute(
            sql.SQL("TRUNCATE TABLE {}").format(sql.Identifier(table.schema, table.table))
        )
    conn.commit()
    print(f"[truncate] {table.schema}.{table.table}")


def ensure_float_int_columns(conn: psycopg.Connection, target: tp.DbTable) -> None:
    q = """
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
          AND column_name = ANY(%s)
    """
    with conn.cursor() as cur:
        cur.execute(q, (target.schema, target.table, tp.INT_COLS))
        rows = cur.fetchall()

    to_alter = [col for col, dtype in rows if dtype != "double precision"]
    if not to_alter:
        print("[schema] int_feature_* already DOUBLE PRECISION")
        return

    with conn.cursor() as cur:
        for col in to_alter:
            cur.execute(
                sql.SQL(
                    "ALTER TABLE {} ALTER COLUMN {} TYPE DOUBLE PRECISION USING {}::DOUBLE PRECISION"
                ).format(
                    sql.Identifier(target.schema, target.table),
                    sql.Identifier(col),
                    sql.Identifier(col),
                )
            )
    conn.commit()
    print(f"[schema] altered_to_double_precision={len(to_alter)}")


def run_log1p_rebuild(
    artifacts: dict[str, Any],
    dsn: str,
    source_table: str,
    target_table: str,
    control_table: str,
    batch_rows: int,
    partition_suffix: str,
    source_partition_like: str,
    add_missing_flags: bool,
    truncate_target: bool,
    drop_indexes: bool,
    recreate_dropped_indexes: bool,
    reset_control_table: bool,
    ensure_float_int_cols: bool,
) -> None:
    source = tp.split_qualified(source_table)
    target = tp.split_qualified(target_table)
    control = tp.split_qualified(control_table)

    with psycopg.connect(dsn) as conn:
        tp.ensure_tables(conn, target, control, add_missing_flags)
        if ensure_float_int_cols:
            ensure_float_int_columns(conn, target)

        captured_indexes: list[tuple[str, str]] = snapshot_target_indexes(conn, target)
        dropped = False
        if drop_indexes:
            drop_target_indexes(conn, target, captured_indexes)
            dropped = True

        if truncate_target:
            truncate_table(conn, target)
        if reset_control_table:
            truncate_table(conn, control)

        all_parts = tp.fetch_source_partitions(conn, source, source_partition_like)
        processed = set() if reset_control_table else tp.get_processed_files(conn, control)
        pending = [p for p in all_parts if p not in processed]
        print(f"source_partitions={len(all_parts)} pending={len(pending)}")

        try:
            for part in pending:
                tp.mark_processing(conn, control, part)
                total_rows = 0
                try:
                    for batch in tp.read_partition_batches(conn, source, part, batch_rows):
                        transformed = transform_batch_log1p(
                            batch,
                            f"{part}{partition_suffix}",
                            artifacts,
                            add_missing_flags,
                        )
                        total_rows += tp.copy_batch(conn, target, transformed, add_missing_flags)
                    tp.mark_processed(conn, control, part, total_rows)
                    print(f"[processed] {part} rows={total_rows}")
                except Exception as exc:
                    conn.rollback()
                    tp.mark_failed(conn, control, part, str(exc))
                    print(f"[failed] {part} error={exc}")
                    raise
        finally:
            if dropped and recreate_dropped_indexes:
                recreate_indexes(conn, captured_indexes)


def main() -> None:
    args = parse_args()
    artifact_path = Path(args.artifact_path)
    add_missing_flags = bool(args.add_missing_flags)
    source_name = args.source_table.strip().lower()
    target_name = args.target_table.strip().lower()

    if source_name == target_name:
        raise ValueError(
            "SOURCE_TABLE and TARGET_TABLE are the same. "
            "Choose a different target table to avoid overwriting source data."
        )

    try:
        artifacts = tp.load_artifacts(artifact_path)
        print(f"[transform-log1p] artifact loaded from {artifact_path}")
    except (FileNotFoundError, ValueError) as exc:
        print(f"[fit] {exc}")
        print("[fit] building artifacts from source table")
        with psycopg.connect(args.pg_dsn) as fit_conn:
            artifacts = tp.fit_artifacts_from_source(
                conn=fit_conn,
                source=tp.split_qualified(args.source_table),
                min_cat_freq=args.min_cat_freq,
                max_cat_levels=args.max_cat_levels,
                source_partition_like=args.source_partition_like,
            )
        tp.save_artifacts(artifact_path, artifacts)
        print(f"[fit] artifact saved to {artifact_path}")

    run_log1p_rebuild(
        artifacts=artifacts,
        dsn=args.pg_dsn,
        source_table=args.source_table,
        target_table=args.target_table,
        control_table=args.control_table,
        batch_rows=args.batch_rows,
        partition_suffix=args.partition_suffix,
        source_partition_like=args.source_partition_like,
        add_missing_flags=add_missing_flags,
        truncate_target=bool(args.truncate_target),
        drop_indexes=bool(args.drop_indexes),
        recreate_dropped_indexes=bool(args.recreate_indexes),
        reset_control_table=bool(args.reset_control_table),
        ensure_float_int_cols=bool(args.ensure_float_int_cols),
    )


if __name__ == "__main__":
    main()
