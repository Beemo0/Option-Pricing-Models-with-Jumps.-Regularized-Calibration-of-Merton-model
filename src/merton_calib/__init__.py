"""
merton_calib — Bibliothèque Python pour la calibration régularisée du modèle de Merton.

PFE MMF ING3 — CY Tech, 2025-2026
Auteurs : Corentin Stephan, Hugo Landron
Encadrants : Y. Aktar, I. Kortchemski
"""

from .simulation import simulate_merton_paths
from .pricing import (
    black_scholes_call,
    black_scholes_put,
    merton_call_analytic,
    merton_call_mc,
)
from .implied_vol import bs_implied_vol
from .calibration import (
    compute_residuals,
    compute_jacobian,
    entropy_KL,
    entropy_grad,
    entropy_hess,
    calibrate_tikhonov,
    calibrate_entropy,
    lcurve,
    synthetic_smile,
)

__all__ = [
    "simulate_merton_paths",
    "black_scholes_call",
    "black_scholes_put",
    "merton_call_analytic",
    "merton_call_mc",
    "bs_implied_vol",
    "compute_residuals",
    "compute_jacobian",
    "entropy_KL",
    "entropy_grad",
    "entropy_hess",
    "calibrate_tikhonov",
    "calibrate_entropy",
    "lcurve",
    "synthetic_smile",
]
