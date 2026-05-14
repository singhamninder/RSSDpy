"""Sampling sub-package: CCD generation, spatial uniformity, and RSSD site selection."""

from rssdpy.sampling.design import central_composite_design
from rssdpy.sampling.rssd import RSSDesign, run_rssd
from rssdpy.sampling.uniformity import average_distance

__all__ = ["central_composite_design", "RSSDesign", "run_rssd", "average_distance"]
