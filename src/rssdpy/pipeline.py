"""End-to-end ESAP-RSSD pipeline: QC validation -> sample plan -> site selection.

Bundles the sequence documented in ``docs/theory/rssd-source-notes.md`` and used by the
demo notebook, the ESAP parity tests, and the Streamlit app, so it is defined once.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from rssdpy.features import ECaPCA, ESAPQCResult, iterative_esap_validation
from rssdpy.sampling import RSSDesign, esap_sample_plan, esap_sampling_design
from rssdpy.sampling.design import DesignMode
from rssdpy.sampling.rssd import run_rssd


@dataclass(frozen=True)
class ESAPRunResult:
    """Bundled result of a full ESAP-RSSD run.

    Attributes
    ----------
    eca_clean : pd.DataFrame
        ECa table after ESAP outlier deletion.
    scores : np.ndarray
        Standardised PC scores aligned to ``eca_clean``.
    coords_clean : np.ndarray
        Coordinates aligned to ``eca_clean`` / ``scores``.
    qc : ESAPQCResult
        ESAP sigma QC result (masking/outlier/eligible masks).
    original_indices : np.ndarray
        Row indices into the original survey for each row of ``eca_clean``.
    design : np.ndarray
        CCD / 2-signal design matrix used for site matching.
    rssd : RSSDesign
        Full RSSD site-selection result.
    """

    eca_clean: pd.DataFrame
    scores: np.ndarray
    coords_clean: np.ndarray
    qc: ESAPQCResult
    original_indices: np.ndarray
    design: np.ndarray
    rssd: RSSDesign


def run_esap_rssd(
    eca: pd.DataFrame,
    coords: np.ndarray,
    *,
    n_components: int = 2,
    target_size: int = 12,
    design_factor: float = 1.0,
    design_mode: DesignMode = "esap_two_signal",
    extra_mode: Literal["global", "cube"] = "global",
    masking_std: float = 3.5,
    outlier_std: float = 4.5,
    opt_criteria_mode: str = "esap",
) -> ESAPRunResult:
    """Run the full ESAP-RSSD workflow: QC validation, sample plan, site selection.

    Parameters
    ----------
    eca : pd.DataFrame
        Raw ECa survey data (positive dS/m), one column per channel.
    coords : np.ndarray
        Shape ``(N, 2)`` projected coordinates (metres), aligned to ``eca`` rows.
    n_components : int
        Number of PCA components. Default 2 (ESAP 2-signal survey).
    target_size : int
        Total RSSD sample size; one of ``(6, 12, 20)``.
    design_factor : float
        ESAP D-Factor multiplier, recommended range 0.90-1.10.
    design_mode : {"ccd", "esap_two_signal"}
        Design generator. Default ``"esap_two_signal"`` for 2-channel surveys.
    extra_mode : {"global", "cube"}
        Extra-site search strategy for extending the core set to ``target_size``.
        Default ``"global"`` — Confirmed, Lesch et al. (1995) Paper 2 steps 6-7 search
        ALL remaining survey sites. Do not pass ``"cube"`` here; it is reserved for the
        separate ``n_validation`` monitoring/validation pass (source notes Section 6).
    masking_std : float
        ESAP masking threshold (sigma). Default 3.5.
    outlier_std : float
        ESAP outlier-deletion threshold (sigma). Default 4.5.
    opt_criteria_mode : {"derived", "esap"}
        Opt-Criteria reporting mode. Default ``"esap"``.

    Returns
    -------
    ESAPRunResult
        Bundled intermediate and final results.

    Raises
    ------
    ValueError
        Propagated from :func:`~rssdpy.features.iterative_esap_validation`,
        :func:`~rssdpy.sampling.esap_sample_plan`, or
        :func:`~rssdpy.sampling.rssd.run_rssd` for invalid inputs.
    """
    pca = ECaPCA(n_components=n_components)
    eca_clean, scores, qc, original_indices = iterative_esap_validation(
        eca, pca, masking_std=masking_std, outlier_std=outlier_std
    )
    coords_clean = np.asarray(coords, dtype=float)[original_indices]

    plan = esap_sample_plan(
        n_components=n_components,
        target_size=target_size,
        design_factor=design_factor,
        design_mode=design_mode,
    )
    design = esap_sampling_design(
        n_components=n_components,
        design_mode=design_mode,
        design_factor=design_factor,
    )
    rssd = run_rssd(
        scores,
        coords_clean,
        design,
        n_extra=plan.n_extra,
        extra_mode=extra_mode,
        eligible_mask=qc.eligible_mask,
        original_indices=original_indices,
        design_factor=design_factor,
        opt_criteria_mode=opt_criteria_mode,
    )
    return ESAPRunResult(
        eca_clean=eca_clean,
        scores=scores,
        coords_clean=coords_clean,
        qc=qc,
        original_indices=original_indices,
        design=design,
        rssd=rssd,
    )
