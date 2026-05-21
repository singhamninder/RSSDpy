"""Log-ECa transformation, standardisation, and PCA.

Theory basis: Lesch, Strauss, and Rhoades (1995) Paper 2 — Confirmed.
  https://doi.org/10.1029/94WR02180

Processing pipeline (source-ordered):
  1. Validate all ECa values are strictly positive (dS/m).
  2. Natural-log transform: ``np.log(eca)``.
  3. Centre and scale each log-ECa column to zero mean and unit variance.
  4. Apply PCA to the scaled matrix.
  5. Standardise each PC score vector by dividing by ``sqrt(eigenvalue)``
     so every score has mean 0 and variance 1.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


class ECaPCA:
    """PCA pipeline for multi-channel ECa survey data.

    Applies the pre-processing chain required by RSSD:
    log-transform → centre/scale → PCA → standardise scores.

    Parameters
    ----------
    n_components : int
        Number of principal components to retain.  For ESAP-compatible
        behaviour with two readings, use 2.  The most common RSSD form
        (Lesch et al. 1995 Paper 2) uses 3.

    Attributes
    ----------
    scaler_ : StandardScaler
        Fitted scaler for log-ECa values (set after :meth:`fit_transform`).
    pca_ : sklearn.decomposition.PCA
        Fitted PCA object (set after :meth:`fit_transform`).
    eigenvalues_ : np.ndarray
        Shape ``(n_components,)`` — variance explained by each component
        (= ``pca_.explained_variance_``).
    eigenvectors_ : np.ndarray
        Shape ``(n_features, n_components)`` — principal component loadings
        (= ``pca_.components_.T``).
    explained_variance_ratio_ : np.ndarray
        Shape ``(n_components,)`` — fraction of total variance per component.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> eca = pd.DataFrame({"EMh": np.exp([1.0, 1.5, 2.0]),
    ...                     "EMv": np.exp([1.2, 1.8, 2.3])})
    >>> pca = ECaPCA(n_components=2)
    >>> scores, evals, _ = pca.fit_transform(eca)
    >>> scores.shape
    (3, 2)
    """

    def __init__(self, n_components: int = 3) -> None:
        if n_components < 1:
            raise ValueError(f"n_components must be ≥ 1, got {n_components}")
        self.n_components = n_components
        self.scaler_: StandardScaler | None = None
        self.pca_: PCA | None = None
        self.eigenvalues_: np.ndarray | None = None
        self.eigenvectors_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None

    def fit_transform(
        self,
        eca: pd.DataFrame,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fit the PCA pipeline and return standardised PC scores.

        Parameters
        ----------
        eca : pd.DataFrame
            ECa readings in dS/m.  Every column is treated as a separate EMI
            channel.  All values must be strictly positive.

        Returns
        -------
        scores : np.ndarray
            Shape ``(N, n_components)``.  Standardised PC scores: each column
            has mean 0 and variance 1 (raw score divided by ``sqrt(eigenvalue)``).
        eigenvalues : np.ndarray
            Shape ``(n_components,)``.
        eigenvectors : np.ndarray
            Shape ``(n_features, n_components)`` — loadings matrix.

        Raises
        ------
        ValueError
            If any ECa value is ≤ 0, or if ``n_components`` exceeds the number
            of ECa channels available.
        """
        if eca.empty:
            raise ValueError("eca DataFrame is empty.")

        n_features = eca.shape[1]
        if self.n_components > n_features:
            raise ValueError(
                f"n_components ({self.n_components}) exceeds the number of ECa "
                f"channels ({n_features})."
            )

        _validate_eca_positive(eca)

        # Step 1 — natural log transform (Confirmed: Paper 2)
        log_eca = np.log(eca.to_numpy(dtype=float))

        # Step 2 — centre and scale (Confirmed: Paper 2)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(log_eca)

        # Step 3 — PCA (Confirmed: Paper 2)
        pca = PCA(n_components=self.n_components)
        raw_scores = pca.fit_transform(scaled)

        # Step 4 — standardise scores by sqrt(eigenvalue) (Confirmed: Paper 2, Lesch 2005)
        eigenvalues = pca.explained_variance_
        scores = raw_scores / np.sqrt(eigenvalues)

        self.scaler_ = scaler
        self.pca_ = pca
        self.eigenvalues_ = eigenvalues
        self.eigenvectors_ = pca.components_.T
        self.explained_variance_ratio_ = pca.explained_variance_ratio_

        logger.debug(
            "ECaPCA fit: n=%d, n_components=%d, explained_var=%s",
            eca.shape[0],
            self.n_components,
            np.round(self.explained_variance_ratio_, 3),
        )
        return scores, eigenvalues, pca.components_.T

    def transform(self, eca: pd.DataFrame) -> np.ndarray:
        """Apply the fitted pipeline to new ECa data.

        Parameters
        ----------
        eca : pd.DataFrame
            ECa readings in dS/m with the same columns used during fitting.
            All values must be strictly positive.

        Returns
        -------
        np.ndarray
            Shape ``(N, n_components)`` — standardised PC scores.

        Raises
        ------
        RuntimeError
            If :meth:`fit_transform` has not been called yet.
        ValueError
            If any ECa value is ≤ 0.
        """
        if self.pca_ is None or self.scaler_ is None or self.eigenvalues_ is None:
            raise RuntimeError("Call fit_transform before transform.")

        _validate_eca_positive(eca)
        log_eca = np.log(eca.to_numpy(dtype=float))
        scaled = self.scaler_.transform(log_eca)
        raw_scores = self.pca_.transform(scaled)
        eigenvalues: np.ndarray = self.eigenvalues_
        return raw_scores / np.sqrt(eigenvalues)


def _validate_eca_positive(eca: pd.DataFrame) -> None:
    """Raise ValueError if any ECa value is non-positive."""
    min_val = eca.to_numpy(dtype=float).min()
    if min_val <= 0:
        raise ValueError(
            f"All ECa values must be strictly positive (dS/m), "
            f"but found minimum value {min_val:.6g}."
        )
