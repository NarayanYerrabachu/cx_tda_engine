import numpy as np
import pandas as pd
import pytest

from cx_tda_engine import DatasetSpec, clear_base_cache, normalized_features

FEATURES = ["f1", "f2", "f3", "f4", "f5"]


def make_df(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Generic table: an id, two label columns, a year and five numeric features
    with a planted outlier block in the first 15 rows."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "record_id": [f"R{i:04d}" for i in range(n)],
        "group_a":   rng.choice(["north", "south", "east"], n),
        "group_b":   rng.choice(["A", "B", "C"], n),
        "year":      rng.choice([2015, 2018, 2021, 2024], n).astype(float),
        "f1":        rng.normal(size=n),
        "f2":        rng.normal(size=n),
        "f3":        rng.lognormal(size=n),
        "f4":        rng.uniform(size=n),
        "f5":        rng.choice([0.0, 1.0], n),
        "metric":    rng.normal(0.1, 0.5, n),
    })
    df.loc[:14, "f3"] += 30.0
    return df


@pytest.fixture
def df() -> pd.DataFrame:
    return make_df()


@pytest.fixture
def spec() -> DatasetSpec:
    return DatasetSpec(id_col="record_id", group_cols=("group_a", "group_b"), time_col="year",
                       display_cols=("group_a", "metric"), stat_cols=("metric",))


@pytest.fixture
def X_norm(df) -> np.ndarray:
    return normalized_features(df, FEATURES)


@pytest.fixture
def record_ids(df) -> list[str]:
    return df["record_id"].astype(str).tolist()


@pytest.fixture(autouse=True)
def _fresh_cache():
    clear_base_cache()
    yield
    clear_base_cache()
