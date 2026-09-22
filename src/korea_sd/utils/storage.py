from __future__ import annotations

import os
from pathlib import Path
import tempfile

import pandas as pd


def atomic_write_parquet(df: pd.DataFrame, path: str | os.PathLike[str]) -> None:
    """Write parquet atomically.

    A temporary file is created in the destination directory and replaces the
    destination only after pandas/pyarrow successfully finishes serialization.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp.parquet", dir=target.parent
    )
    os.close(fd)
    try:
        df.to_parquet(tmp_name, index=False)
        os.replace(tmp_name, target)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
