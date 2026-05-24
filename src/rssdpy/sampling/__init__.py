"""Sampling sub-package: CCD generation, spatial uniformity, and RSSD site selection."""

from rssdpy.sampling.design import (
    ESAPSamplePlan,
    central_composite_design,
    esap_sample_plan,
    esap_sampling_design,
    esap_two_signal_design,
)
from rssdpy.sampling.rssd import RSSDesign, run_rssd
from rssdpy.sampling.uniformity import (
    average_distance,
    characteristic_spacing,
    compute_opt_criteria,
    opt_criteria_esap,
    opt_criteria_from_ad,
)

__all__ = [
    "central_composite_design",
    "esap_sample_plan",
    "esap_sampling_design",
    "esap_two_signal_design",
    "ESAPSamplePlan",
    "RSSDesign",
    "run_rssd",
    "average_distance",
    "characteristic_spacing",
    "compute_opt_criteria",
    "opt_criteria_esap",
    "opt_criteria_from_ad",
]
