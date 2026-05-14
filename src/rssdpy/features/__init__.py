"""Features sub-package: ECa log-transform, PCA, and outlier detection."""

from rssdpy.features.outliers import detect_outliers
from rssdpy.features.pca import ECaPCA

__all__ = ["ECaPCA", "detect_outliers"]
