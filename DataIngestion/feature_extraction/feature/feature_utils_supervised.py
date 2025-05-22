"""Supervised feature selection helpers using tsfresh."""

from __future__ import annotations

import pandas as pd

from tsfresh.feature_selection import select_features
from tsfresh.utilities.dataframe_functions import impute


def select_tsfresh_features(
    X: pd.DataFrame,
    y: pd.Series,
    fdr: float = 0.05,
    n_jobs: int = 0,
    chunksize: int | None = None,
) -> pd.DataFrame:
    """Return features selected using tsfresh's supervised FRESH algorithm.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix with observations as rows.
    y : pd.Series
        Target vector aligned with ``X``.
    fdr : float, default 0.05
        False discovery rate level for hypothesis tests.
    n_jobs : int, default 0
        Number of parallel workers. ``0`` means use all cores.
    chunksize : int | None, optional
        Optional chunk size for distributed execution.
    """
    if X.empty:
        return X
    X_imp = impute(X.copy())
    return select_features(
        X_imp, y, fdr_level=fdr, n_jobs=n_jobs, chunksize=chunksize
    )

