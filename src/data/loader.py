"""
Universal Data Loader — supports multiple formats and sources.

Loads data from:
- Local files (CSV, Excel, Parquet, JSON, Feather, ORC)
- Cloud storage via fsspec (S3, GCS, Azure Blob, etc.)
- Databases via SQLAlchemy (PostgreSQL, MySQL, SQLite, etc.)

Auto-detects format from file extension or explicit format parameter.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Union
from urllib.parse import urlparse

import pandas as pd

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {
    ".csv": "csv",
    ".tsv": "csv",
    ".txt": "csv",
    ".parquet": "parquet",
    ".pq": "parquet",
    ".xlsx": "excel",
    ".xls": "excel",
    ".json": "json",
    ".jsonl": "json",
    ".feather": "feather",
    ".ftr": "feather",
    ".orc": "orc",
}


class DataSourceConfig:
    """Configuration for a data source."""

    def __init__(
        self,
        source: str,
        format: Optional[str] = None,
        sql_query: Optional[str] = None,
        sql_params: Optional[dict] = None,
        chunk_size: Optional[int] = None,
        read_kwargs: Optional[dict] = None,
    ) -> None:
        self.source = source
        self.format = format
        self.sql_query = sql_query
        self.sql_params = sql_params
        self.chunk_size = chunk_size
        self.read_kwargs = read_kwargs or {}

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "DataSourceConfig":
        """Create config from dictionary (e.g., DAG params)."""
        # Extract read_kwargs separately to avoid double-wrapping
        read_kwargs = config.get("read_kwargs", {})
        return cls(
            source=config.get("source", ""),
            format=config.get("format"),
            sql_query=config.get("sql_query"),
            sql_params=config.get("sql_params"),
            chunk_size=config.get("chunk_size"),
            read_kwargs=read_kwargs,
        )


def _infer_format(source: str, explicit_format: Optional[str] = None) -> str:
    """Infer file format from extension or use explicit format."""
    if explicit_format:
        return explicit_format.lower()

    parsed = urlparse(source)
    path = parsed.path if parsed.scheme else source
    ext = Path(path).suffix.lower()

    if ext in SUPPORTED_FORMATS:
        return SUPPORTED_FORMATS[ext]

    # Default to parquet
    logger.warning(f"Could not infer format for {source}, defaulting to parquet")
    return "parquet"


def _is_cloud_path(source: str) -> bool:
    """Check if source is a cloud storage path."""
    parsed = urlparse(source)
    return parsed.scheme in {"s3", "gs", "gcs", "az", "abfs", "abfss", "hdfs", "webhdfs"}


def _is_database_url(source: str) -> bool:
    """Check if source is a database connection URL."""
    parsed = urlparse(source)
    return parsed.scheme in {
        "postgresql",
        "postgres",
        "mysql",
        "sqlite",
        "mssql",
        "oracle",
        "redshift",
        "snowflake",
        "bigquery",
    }


def load_data(config: DataSourceConfig) -> pd.DataFrame:
    """
    Load data from various sources and formats.

    Args:
        config: DataSourceConfig with source, format, and read options.

    Returns:
        DataFrame with loaded data.

    Examples:
        # Local CSV
        load_data(DataSourceConfig(source="/data/file.csv"))

        # S3 Parquet
        load_data(DataSourceConfig(source="s3://bucket/data.parquet"))

        # PostgreSQL
        load_data(DataSourceConfig(
            source="postgresql://user:pass@host/db",
            sql_query="SELECT * FROM features WHERE dt = '2024-01-01'"
        ))

        # Excel with specific sheet
        load_data(DataSourceConfig(
            source="/data/file.xlsx",
            read_kwargs={"sheet_name": "Sheet1"}
        ))
    """
    source = config.source
    fmt = _infer_format(source, config.format)

    logger.info(f"Loading data from {source} (format: {fmt})")

    # Database source
    if _is_database_url(source) or config.sql_query:
        return _load_from_database(config)

    # Cloud storage or local file
    return _load_from_file(config, fmt)


def _load_from_file(config: DataSourceConfig, fmt: str) -> pd.DataFrame:
    """Load data from file (local or cloud via fsspec)."""
    source = config.source
    read_kwargs = config.read_kwargs.copy()

    # Handle chunked reading
    if config.chunk_size:   
        read_kwargs["chunksize"] = config.chunk_size
        chunks = _read_chunks(source, fmt, read_kwargs)
        return pd.concat(chunks, ignore_index=True)

    # Single read
    return _read_single(source, fmt, read_kwargs)


def _read_single(source: str, fmt: str, read_kwargs: dict) -> pd.DataFrame:
    """Read a single file based on format."""
    readers = {
        "csv": pd.read_csv,
        "parquet": pd.read_parquet,
        "excel": pd.read_excel,
        "json": pd.read_json,
        "feather": pd.read_feather,
        "orc": pd.read_orc,
    }

    if fmt not in readers:
        raise ValueError(f"Unsupported format: {fmt}. Supported: {list(readers.keys())}")

    reader = readers[fmt]

    # Format-specific defaults
    if fmt == "csv":
        read_kwargs.setdefault("sep", read_kwargs.pop("sep", ","))
        read_kwargs.setdefault("encoding", read_kwargs.pop("encoding", "utf-8"))
    elif fmt == "excel":
        read_kwargs.setdefault("engine", read_kwargs.pop("engine", "openpyxl"))
    elif fmt == "parquet":
        read_kwargs.setdefault("engine", read_kwargs.pop("engine", "pyarrow"))

    # Cloud storage paths work directly with fsspec
    return reader(source, **read_kwargs)


def _read_chunks(source: str, fmt: str, read_kwargs: dict) -> list[pd.DataFrame]:
    """Read file in chunks."""
    chunks = []
    # Only CSV supports chunksize natively in pandas
    if fmt == "csv":
        reader = pd.read_csv(source, **read_kwargs)
    else:
        # For other formats, read full and split manually (fallback)
        df = _read_single(source, fmt, read_kwargs)
        chunk_size = read_kwargs.get("chunksize", 10000)
        for i in range(0, len(df), chunk_size):
            chunks.append(df.iloc[i:i + chunk_size])
        return chunks
    
    for chunk in reader:
        chunks.append(chunk)
    return chunks


def _load_from_database(config: DataSourceConfig) -> pd.DataFrame:
    """Load data from database using SQLAlchemy."""
    from sqlalchemy import create_engine, text

    source = config.source
    sql_query = config.sql_query

    if not sql_query:
        raise ValueError("sql_query is required for database sources")

    engine = create_engine(source)

    read_kwargs = config.read_kwargs.copy()
    params = config.sql_params or {}

    if config.chunk_size:
        read_kwargs["chunksize"] = config.chunk_size
        chunks = pd.read_sql(sql_query, engine, params=params, **read_kwargs)
        return pd.concat(chunks, ignore_index=True)

    return pd.read_sql(sql_query, engine, params=params, **read_kwargs)


def load_production_data(config: DataSourceConfig) -> pd.DataFrame:
    """Load production data with validation."""
    df = load_data(config)
    logger.info(f"Loaded production data: {len(df)} rows, {len(df.columns)} columns")
    return df


def load_reference_data(config: DataSourceConfig) -> pd.DataFrame:
    """Load reference data with validation."""
    df = load_data(config)
    logger.info(f"Loaded reference data: {len(df)} rows, {len(df.columns)} columns")
    return df


def save_data(
    df: pd.DataFrame,
    path: str,
    format: Optional[str] = None,
    **write_kwargs: Any,
) -> str:
    """
    Save DataFrame to various formats.

    Args:
        df: DataFrame to save
        path: Destination path (local or cloud)
        format: Output format (inferred from extension if not provided)
        **write_kwargs: Additional arguments for writer

    Returns:
        The path where data was saved.
    """
    fmt = _infer_format(path, format)

    writers = {
        "csv": df.to_csv,
        "parquet": df.to_parquet,
        "excel": df.to_excel,
        "json": df.to_json,
        "feather": df.to_feather,
        "orc": lambda p, **kw: df.to_orc(p, **kw),
    }

    if fmt not in writers:
        raise ValueError(f"Unsupported write format: {fmt}")

    # Format-specific defaults
    if fmt == "csv":
        write_kwargs.setdefault("index", False)
        write_kwargs.setdefault("encoding", "utf-8")
    elif fmt == "parquet":
        write_kwargs.setdefault("index", False)
        write_kwargs.setdefault("engine", "pyarrow")
    elif fmt == "excel":
        write_kwargs.setdefault("index", False)
        write_kwargs.setdefault("engine", "openpyxl")

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    writers[fmt](path, **write_kwargs)

    logger.info(f"Saved {len(df)} rows to {path} (format: {fmt})")
    return path