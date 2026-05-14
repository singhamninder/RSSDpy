"""Shared pytest fixtures for RSSDpy tests.

All synthetic data here is generated deterministically from a fixed seed so
tests are reproducible without any external files.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """Seeded random generator used across all test modules."""
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def synthetic_eca_df(rng: np.random.Generator) -> pd.DataFrame:
    """300-site synthetic ECa survey with two readings (EMh, EMv) in dS/m.

    Values are log-normally distributed to mimic real ECa surveys.
    EMv is correlated with EMh (deeper reading tends to track shallow).

    Returns
    -------
    pd.DataFrame
        Columns: ``EMh``, ``EMv`` — positive floats in dS/m.
    """
    n = 300
    emh = np.exp(rng.normal(loc=1.5, scale=0.6, size=n))  # median ≈ 4.5 dS/m
    noise = rng.normal(0, 0.3, size=n)
    emv = np.exp(np.log(emh) * 0.9 + 0.3 + noise)
    return pd.DataFrame({"EMh": emh, "EMv": emv})


@pytest.fixture(scope="session")
def synthetic_coords(rng: np.random.Generator) -> np.ndarray:
    """300-site geographic coordinates in projected metres (UTM-like).

    Returns
    -------
    np.ndarray
        Shape (300, 2), columns [easting_m, northing_m].
    """
    n = 300
    x = rng.uniform(0, 1000, size=n)
    y = rng.uniform(0, 500, size=n)
    return np.column_stack([x, y])


@pytest.fixture(scope="session")
def synthetic_ece(rng: np.random.Generator, synthetic_eca_df: pd.DataFrame) -> np.ndarray:
    """Synthetic ln(ECe) values at a 16-site calibration subset.

    ECe is generated as a noisy linear function of ln(ECa) to mimic real
    calibration data.  The first 16 sites in ``synthetic_eca_df`` are used.

    Returns
    -------
    np.ndarray
        Shape (16,) — natural log of ECe in dS/m.
    """
    n_cal = 16
    ln_emh = np.log(synthetic_eca_df["EMh"].to_numpy()[:n_cal])
    ln_ece = 0.8 * ln_emh + 0.5 + rng.normal(0, 0.2, size=n_cal)
    return ln_ece
