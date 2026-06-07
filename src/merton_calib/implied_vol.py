"""
implied_vol.py — Inversion de la formule Black-Scholes pour la volatilité implicite.

Utilise la méthode de Brent (scipy.optimize.brentq) pour résoudre :
    C_BS(S0, K, T, r, sigma_imp) = C_market

References
----------
Black, F. & Scholes, M. (1973). The pricing of options and corporate liabilities.
    Journal of Political Economy, 81(3), 637-654.
"""

from __future__ import annotations

import numpy as np

from .pricing import black_scholes_call, bs_vega


def _brentq(f, a: float, b: float, xtol: float = 1e-8, maxiter: int = 100) -> float:
    """
    Méthode de Brent pour trouver la racine de f sur [a, b].

    Implémentation pure NumPy (sans scipy) basée sur l'algorithme de Brent (1973).
    Hypothèse : f(a) * f(b) < 0.
    """
    fa, fb = f(a), f(b)
    if fa * fb > 0:
        raise ValueError("f(a) et f(b) doivent être de signes opposés.")

    c, fc = b, fb
    d = e = b - a

    for _ in range(maxiter):
        if fb * fc > 0:
            c, fc = a, fa
            d = e = b - a

        if abs(fc) < abs(fb):
            a, fa = b, fb
            b, fb = c, fc
            c, fc = a, fa

        tol = 2.0 * xtol * abs(b) + 0.5 * xtol
        xm = 0.5 * (c - b)

        if abs(xm) <= tol or fb == 0.0:
            return b

        if abs(e) >= tol and abs(fa) > abs(fb):
            s = fb / fa
            if a == c:
                p = 2.0 * xm * s
                q = 1.0 - s
            else:
                q = fa / fc
                r = fb / fc
                p = s * (2.0 * xm * q * (q - r) - (b - a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)

            if p > 0:
                q = -q
            else:
                p = -p

            if 2.0 * p < min(3.0 * xm * q - abs(tol * q), abs(e * q)):
                e, d = d, p / q
            else:
                d = e = xm
        else:
            d = e = xm

        a, fa = b, fb

        if abs(d) > tol:
            b += d
        else:
            b += tol if xm > 0 else -tol

        fb = f(b)

    return b  # Retourne la meilleure estimation même si non convergé


def bs_implied_vol(
    C_market: float,
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma_min: float = 1e-6,
    sigma_max: float = 10.0,
    tol: float = 1e-8,
) -> float:
    """
    Calcule la volatilité implicite Black-Scholes par inversion numérique (méthode de Brent).

    Résout : C_BS(S0, K, T, r, sigma) = C_market

    Parameters
    ----------
    C_market : float
        Prix de marché observé du call européen.
    S0 : float
        Prix initial du sous-jacent.
    K : float
        Strike de l'option.
    T : float
        Maturité (en années).
    r : float
        Taux sans risque continu.
    sigma_min : float, optional
        Borne inférieure de la recherche. Défaut : 1e-6.
    sigma_max : float, optional
        Borne supérieure de la recherche. Défaut : 10.0 (1000%).
    tol : float, optional
        Tolérance de convergence pour l'algorithme de Brent. Défaut : 1e-8.

    Returns
    -------
    float
        Volatilité implicite sigma_imp > 0, ou np.nan si l'inversion échoue.

    Notes
    -----
    L'algorithme de Brent est garanti à converger pour une fonction continue
    changeant de signe sur [sigma_min, sigma_max]. La condition nécessaire est que
    le prix de marché soit compris entre les valeurs intrinsèques extrêmes :
        max(S0 - K*exp(-rT), 0) < C_market < S0.
    Voir Cont & Tankov (2004, Ch. 11) pour l'usage dans le contexte de calibration.
    """
    # Valeur intrinsèque (borne inférieure sur le prix du call)
    intrinsic = max(S0 - K * np.exp(-r * T), 0.0)

    # Vérification des bornes de no-arbitrage
    if C_market <= intrinsic or C_market >= S0:
        return np.nan

    # Fonction objectif
    def objective(sigma: float) -> float:
        return black_scholes_call(S0, K, T, r, sigma) - C_market

    # Vérification que l'objectif change de signe sur [sigma_min, sigma_max]
    f_min = objective(sigma_min)
    f_max = objective(sigma_max)

    if f_min * f_max > 0:
        return np.nan

    try:
        sigma_imp = _brentq(objective, sigma_min, sigma_max, xtol=tol)
        return float(sigma_imp)
    except (ValueError, RuntimeError):
        return np.nan


def merton_implied_vol(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    lambda_: float,
    mu_J: float,
    delta: float,
    n_terms: int = 50,
) -> float:
    """
    Calcule la volatilité implicite Black-Scholes correspondant au prix Merton analytique.

    Étape 1 : Calcule C_Merton = merton_call_analytic(...)
    Étape 2 : Inverse C_BS(sigma_imp) = C_Merton via bs_implied_vol.

    Parameters
    ----------
    S0 : float
    K : float
    T : float
    r : float
    sigma : float
        Volatilité diffusive du modèle de Merton.
    lambda_ : float
    mu_J : float
    delta : float
    n_terms : int, optional

    Returns
    -------
    float
        Volatilité implicite Black-Scholes, ou np.nan si l'inversion échoue.

    Notes
    -----
    C'est cette fonction qui permet de tracer le smile de volatilité du modèle de Merton.
    """
    from .pricing import merton_call_analytic

    price = merton_call_analytic(S0, K, T, r, sigma, lambda_, mu_J, delta, n_terms)
    return bs_implied_vol(price, S0, K, T, r)
