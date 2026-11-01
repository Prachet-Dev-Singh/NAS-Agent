"""
hypervolume.py
--------------
Multi-objective Pareto dominance and exact hypervolume calculation (Blueprint Section 8.1 & 9).

Objectives:
1. val_acc (maximize)
2. params_m (minimize, bounded by params_cap_m)
3. macs_g (minimize, bounded by macs_cap_g)

Conventions:
- In normalized space, we define positive headroom objectives to maximize against reference origin (0, 0, 0):
    a = clip(val_acc, 0, 1)
    p = clip(1.0 - params_m / params_cap_m, 0, 1)
    m = clip(1.0 - macs_g / macs_cap_g, 0, 1)
- Infeasible models (params > cap or macs > cap) have headroom = 0 and contribute 0 hypervolume.
- Monotonicity: Hypervolume of an archive of non-dominated feasible points monotonically increases.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple
import numpy as np


def dominates(
    p1: Tuple[float, float, float],
    p2: Tuple[float, float, float],
) -> bool:
    """True if point p1 dominates p2 in 3D maximization space."""
    better_or_equal = (p1[0] >= p2[0]) and (p1[1] >= p2[1]) and (p1[2] >= p2[2])
    strictly_better = (p1[0] > p2[0]) or (p1[1] > p2[1]) or (p1[2] > p2[2])
    return better_or_equal and strictly_better


def filter_non_dominated_3d(points: Sequence[Tuple[float, float, float]]) -> List[Tuple[float, float, float]]:
    """Filter a set of 3D points (a, p, m) to retain only the non-dominated Pareto front."""
    if not points:
        return []

    # Deduplicate and sort
    pts = sorted(list(set(points)), key=lambda x: (x[0], x[1], x[2]), reverse=True)
    front: List[Tuple[float, float, float]] = []

    for pt in pts:
        is_dominated = False
        for f in front:
            if dominates(f, pt):
                is_dominated = True
                break
        if not is_dominated:
            # Remove any existing point in front dominated by current pt
            front = [f for f in front if not dominates(pt, f)]
            front.append(pt)

    return front


def compute_hypervolume_2d(points_2d: Sequence[Tuple[float, float]]) -> float:
    """Exact 2D hypervolume calculation with reference point (0, 0).

    points_2d : list of (x, y) coordinates to maximize in [0, 1]^2.
    """
    if not points_2d:
        return 0.0

    # Filter dominated points
    pts = sorted(list(set(points_2d)), key=lambda pt: pt[0], reverse=True)
    clean_pts: List[Tuple[float, float]] = []
    max_y = -1.0
    for x, y in pts:
        if y > max_y:
            clean_pts.append((x, y))
            max_y = y

    clean_pts.sort(key=lambda pt: pt[0])  # ascending x
    hv = 0.0
    prev_x = 0.0
    for i, (x, y) in enumerate(clean_pts):
        width = x - prev_x
        # Height is max y among remaining points
        max_remaining_y = max(p[1] for p in clean_pts[i:])
        hv += width * max_remaining_y
        prev_x = x

    return max(0.0, float(hv))


def compute_hypervolume_3d(
    points_3d: Sequence[Tuple[float, float, float]],
) -> float:
    """Exact 3D hypervolume calculation with reference point (0, 0, 0).

    Uses depth-slice integration along the first coordinate (val_acc).
    """
    front = filter_non_dominated_3d(points_3d)
    if not front:
        return 0.0

    # Sort front by first coordinate (z / val_acc) ascending
    front.sort(key=lambda pt: pt[0])
    z_coords = sorted(list(set(pt[0] for pt in front)))

    total_volume = 0.0
    prev_z = 0.0

    for z in z_coords:
        slice_thickness = z - prev_z
        if slice_thickness <= 0.0:
            continue

        # Points that extend to at least z
        active_2d = [(pt[1], pt[2]) for pt in front if pt[0] >= z]
        area_2d = compute_hypervolume_2d(active_2d)
        total_volume += slice_thickness * area_2d
        prev_z = z

    return max(0.0, float(total_volume))


def calculate_normalized_hypervolume(
    candidates: Sequence[Dict[str, Any]],
    params_cap_m: float,
    macs_cap_g: float,
) -> float:
    """Compute exact normalized hypervolume from a list of candidate dictionaries.

    Each candidate dict must have:
      - 'val_acc' (float)
      - 'params_m' (float)
      - 'macs_g' (float)
      - 'status' (optional, must be 'feasible' or 'ok')
    """
    valid_points: List[Tuple[float, float, float]] = []

    for c in candidates:
        status = c.get("status")
        val_acc = c.get("val_acc")
        params_m = c.get("params_m")
        macs_g = c.get("macs_g")

        if val_acc is None or params_m is None or macs_g is None:
            continue
        if status not in (None, "ok", "feasible"):
            continue

        # Infeasible checks
        if params_m > params_cap_m or macs_g > macs_cap_g:
            continue

        a = min(1.0, max(0.0, float(val_acc)))
        p = min(1.0, max(0.0, 1.0 - float(params_m) / params_cap_m))
        m = min(1.0, max(0.0, 1.0 - float(macs_g) / macs_cap_g))

        if a > 0.0 and p > 0.0 and m > 0.0:
            valid_points.append((a, p, m))

    return compute_hypervolume_3d(valid_points)
