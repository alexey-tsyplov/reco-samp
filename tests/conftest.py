from pathlib import Path

import polars as pl
import pytest
from polars import DataFrame


DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def interactions() -> DataFrame:
    return pl.read_csv(DATA_DIR / "movie_ratings.csv")
