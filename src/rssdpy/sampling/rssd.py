"""Full RSSD site-selection pipeline.

Theory basis: Lesch, Strauss, and Rhoades (1995) Paper 2 — Confirmed.
  https://doi.org/10.1029/94WR02180

Algorithm steps (Paper 2 order):
  1. Compute DLS (Design Level Similarity) = squared Euclidean distance from
     each survey site to each CCD design level in standardised PC space.
  2. For each design level, rank survey sites by DLS → candidate sets
     ψ₁ (best), ψ₂, ψ₃.  Uniqueness is enforced: once a site is assigned
     to a ψ set it cannot appear in any other.
  3. Initialise β = ψ₁ (best statistical match).
  4. Iterative swapping: for each design level try ψ₂ replacement; keep if
     AD decreases.  Repeat with ψ₃.  Loop until no swap improves AD.
  5. Greedy extra-site additions: one at a time from remaining N − n sites,
     accept the site that maximally reduces AD.
  6. Validation sites: rerun Steps 1–5 on remaining sites using only the
     cube design levels.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from rssdpy.sampling.uniformity import average_distance

logger = logging.getLogger(__name__)


@dataclass
class RSSDesign:
    """Result of the RSSD site-selection algorithm.

    Attributes
    ----------
    selected_indices : np.ndarray
        Integer indices of the selected calibration sites in the original
        survey array.  Length = number of CCD levels + n_extra.
    design_level_indices : np.ndarray
        Integer indices of sites matched to the primary ψ₁ design levels,
        in design-level order.  Length = number of CCD levels.
    dls_values : np.ndarray
        DLS score for each primary (ψ₁) match.  Shape ``(n_levels,)``.
    ad_initial : float
        AD(β) before any swapping, using only ψ₁ matches.
    ad_final : float
        AD(β) after swapping and extra-site additions.
    ad_trace : list[float]
        AD value after each swap iteration and each extra-site addition.
    extra_indices : np.ndarray
        Integer indices of the extra sites added after swapping.
    swap_count : int
        Total number of accepted swaps.
    """

    selected_indices: np.ndarray
    design_level_indices: np.ndarray
    dls_values: np.ndarray
    ad_initial: float
    ad_final: float
    ad_trace: list[float] = field(default_factory=list)
    extra_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    swap_count: int = 0


def _build_candidate_sets(
    scores: np.ndarray,
    design: np.ndarray,
    n_candidates: int = 3,
) -> tuple[list[list[int]], np.ndarray]:
    """Find the top-n_candidates survey sites per design level by DLS.

    Uniqueness is enforced globally: once a site is assigned to a candidate
    set, it is removed from consideration for all subsequent design levels.

    Parameters
    ----------
    scores : np.ndarray
        Shape ``(N, n_components)`` — standardised PC scores.
    design : np.ndarray
        Shape ``(n_levels, n_components)`` — CCD design levels.
    n_candidates : int
        Number of candidates per design level (ψ₁ … ψ_k).

    Returns
    -------
    candidates : list[list[int]]
        ``candidates[j]`` is a list of site indices for design level j,
        ordered from best (ψ₁) to worst match.  Each index appears at most
        once across all sets.
    dls_matrix : np.ndarray
        Shape ``(N, n_levels)`` — pairwise DLS values (before uniqueness
        enforcement; useful for diagnostics).
    """
    _n_sites, n_levels = len(scores), len(design)

    # DLS = squared Euclidean distance in PC space (Confirmed: Paper 2)
    # Shape (N, n_levels)
    diff = scores[:, np.newaxis, :] - design[np.newaxis, :, :]
    dls_matrix = np.sum(diff**2, axis=2)

    assigned: set[int] = set()
    candidates: list[list[int]] = []

    for j in range(n_levels):
        sorted_sites = np.argsort(dls_matrix[:, j])
        level_candidates: list[int] = []
        for site_idx in sorted_sites:
            if len(level_candidates) == n_candidates:
                break
            if int(site_idx) not in assigned:
                level_candidates.append(int(site_idx))
                assigned.add(int(site_idx))
        candidates.append(level_candidates)

    return candidates, dls_matrix


def run_rssd(
    scores: np.ndarray,
    coords: np.ndarray,
    design: np.ndarray,
    n_extra: int = 0,
    n_candidates: int = 3,
) -> RSSDesign:
    """Run the full RSSD site-selection algorithm.

    Parameters
    ----------
    scores : np.ndarray
        Shape ``(N, n_components)`` — standardised PC scores for all survey
        sites (outliers should already be removed).
    coords : np.ndarray
        Shape ``(N, 2)`` — projected geographic coordinates in metres.
    design : np.ndarray
        Shape ``(n_levels, n_components)`` — CCD design matrix, typically
        from :func:`~rssdpy.sampling.design.central_composite_design`.
    n_extra : int
        Number of extra sites to add beyond the core n_levels after swapping.
        Extra sites are chosen greedily to minimise AD.  Default 0.
    n_candidates : int
        Number of candidate sites per design level (ψ₁, ψ₂, ψ₃, …).
        Must be ≥ 1.  Default 3 matches Paper 2.

    Returns
    -------
    RSSDesign
        Structured result object; see :class:`RSSDesign`.

    Raises
    ------
    ValueError
        If ``scores`` and ``coords`` have incompatible lengths, if ``design``
        has more components than ``scores``, or if ``n_candidates < 1``.

    Notes
    -----
    The uniqueness constraint (no site in more than one ψ set) is enforced
    during candidate construction, not during swapping.  Swapping only
    considers the pre-built ψ₂ and ψ₃ slots; a swap replaces the current
    assignment for one design level with the next available candidate.
    """
    scores = np.asarray(scores, dtype=float)
    coords = np.asarray(coords, dtype=float)
    design = np.asarray(design, dtype=float)

    n_sites = len(scores)
    n_levels = len(design)

    if len(coords) != n_sites:
        raise ValueError(f"scores has {n_sites} rows but coords has {len(coords)} rows.")
    if design.shape[1] != scores.shape[1]:
        raise ValueError(
            f"design has {design.shape[1]} components but scores has {scores.shape[1]}."
        )
    if n_candidates < 1:
        raise ValueError(f"n_candidates must be ≥ 1, got {n_candidates}")
    if n_sites < n_levels:
        raise ValueError(
            f"Need at least {n_levels} survey sites to fill {n_levels} design levels, "
            f"but only {n_sites} sites provided."
        )

    # Step 1–2: build candidate sets ψ₁, ψ₂, ψ₃ (Confirmed: Paper 2)
    candidates, dls_matrix = _build_candidate_sets(scores, design, n_candidates)

    # Initialise β = ψ₁ (Confirmed: Paper 2)
    current_assignment = [cands[0] for cands in candidates]
    beta = np.array(current_assignment, dtype=int)

    ad_initial = average_distance(coords, beta)
    ad_trace: list[float] = [ad_initial]
    swap_count = 0

    # Step 4: iterative swapping (Confirmed: Paper 2)
    improved = True
    while improved:
        improved = False
        for j in range(n_levels):
            # try ψ₂, ψ₃, … in order
            for candidate_rank in range(1, n_candidates):
                if candidate_rank >= len(candidates[j]):
                    break
                new_site = candidates[j][candidate_rank]
                old_site = beta[j]
                beta[j] = new_site
                new_ad = average_distance(coords, beta)
                if new_ad < ad_trace[-1]:
                    # Accept swap
                    ad_trace.append(new_ad)
                    swap_count += 1
                    improved = True
                    break  # move to next design level after accepting
                else:
                    # Revert
                    beta[j] = old_site

    dls_values = np.array([dls_matrix[beta[j], j] for j in range(n_levels)])

    # Step 5: greedy extra sites (Confirmed: Paper 2)
    used = set(beta.tolist())
    remaining = np.array([i for i in range(n_sites) if i not in used], dtype=int)
    extra_indices: list[int] = []

    for _ in range(n_extra):
        if len(remaining) == 0:
            logger.warning("No remaining sites to add as extra; stopping early.")
            break
        best_ad = ad_trace[-1]
        best_site: int | None = None
        for candidate_idx in range(len(remaining)):
            site = remaining[candidate_idx]
            trial_beta = np.concatenate([beta, [site]])
            trial_ad = average_distance(coords, trial_beta)
            if trial_ad < best_ad:
                best_ad = trial_ad
                best_site = site
        if best_site is None:
            logger.warning(
                "No extra site further reduced AD; stopping extra additions after %d.",
                len(extra_indices),
            )
            break
        extra_indices.append(best_site)
        beta = np.append(beta, best_site)
        remaining = remaining[remaining != best_site]
        ad_trace.append(best_ad)

    ad_final = ad_trace[-1]
    logger.info(
        "RSSD complete: %d core + %d extra sites, AD %.2f → %.2f, %d swaps.",
        n_levels,
        len(extra_indices),
        ad_initial,
        ad_final,
        swap_count,
    )

    return RSSDesign(
        selected_indices=beta,
        design_level_indices=np.array(current_assignment, dtype=int),
        dls_values=dls_values,
        ad_initial=ad_initial,
        ad_final=ad_final,
        ad_trace=ad_trace,
        extra_indices=np.array(extra_indices, dtype=int),
        swap_count=swap_count,
    )
