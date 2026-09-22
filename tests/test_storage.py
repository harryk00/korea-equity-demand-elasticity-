from pathlib import Path

import pandas as pd
import pytest

from korea_sd.utils.storage import atomic_write_parquet


def test_atomic_parquet(tmp_path):
    pytest.importorskip("pyarrow")
    path = tmp_path / "x.parquet"
    atomic_write_parquet(pd.DataFrame({"a": [1, 2]}), path)
    assert path.exists()
    assert pd.read_parquet(path)["a"].tolist() == [1, 2]
