"""
pricing.py — Formule analytique de Merton et pricing Monte-Carlo.

Trois méthodes de valorisation d'une option call européenne dans le modèle de Merton :
  1. Formule analytique (série de Black-Scholes pondérée par Poisson).
  2. Monte-Carlo brut.
  3. Monte-Carlo avec variable de contrôle Black-Scholes.

References
----------
Merton, R.C. (1976). Option pricing when underlying stock returns are discontinuous.
    Journal of Financial Economics, 3(1-2), 125-144.
Cont, R. & Tankov, P. (2004). Financial Modelling with Jump Processes.
    CRC Press. (Ch. 11-12)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


import math as _math


def _norm_cdf(x: float | np.ndarray) -> float | np.ndarray:
    """Fonction de répartition de la loi normale standard via math.erf (stdlib)."""
    if np.ndim(x) == 0:
        return 0.5 * (1.0 + _math.erf(float(x) / _math.sqrt(2.0)))
    return np.array([0.5 * (1.0 + _math.erf(float(v) / _math.sqrt(2.0))) for v in np.asarray(x).flat]).reshape(np.asarray(x).shape)


def _norm_pdf(x: float | np.ndarray) -> float | np.ndarray:
    """Densité de la loi normale standard."""
    return np.exp(-0.5 * np.asarray(x, dtype=float)**2) / _math.sqrt(2.0 * _math.pi)


# Classe proxy pour l'interface norm.cdf / norm.pdf
class _Norm:
    cdf = staticmethod(_norm_cdf)
    pdf = staticmethod(_norm_pdf)

norm = _Norm()


# ─────────────────────────────────────────────────────────────────────────────
# Black-Scholes
# ─────────────────────────────────────────────────────────────────────────────


def black_scholes_call(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
) -> float:
    """
    Prix d'un call européen par la formule de Black-Scholes (1973).

        C_BS = S0 * Phi(d1) - K * exp(-rT) * Phi(d2)

    avec d1 = [log(S0/K) + (r + sigma^2/2)*T] / (sigma*sqrt(T))
         d2 = d1 - sigma*sqrt(T)

    Parameters
    ----------
    S0 : float
        Prix initial du sous-jacent.
    K : float
        Strike de l'option.
    T : float
        Maturité (en années).
    r : float
        Taux sans risque continu.
    sigma : float
        Volatilité (> 0).

    Returns
    -------
    float
        Prix du call européen.

    Notes
    -----
    Si T = 0, retourne max(S0 - K, 0) (valeur intrinsèque).
    Si sigma = 0, retourne max(S0*exp(-qT) - K*exp(-rT), 0) (pas de dividende ici).
    """
    if T <= 0.0:
        return float(max(S0 - K, 0.0))
    if sigma <= 0.0:
        return float(max(S0 - K * np.exp(-r * T), 0.0))

    sqrt_T = np.sqrt(T)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return float(S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


def black_scholes_put(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
) -> float:
    """
    Prix d'un put européen par parité call-put.

        P_BS = C_BS - S0 + K * exp(-rT)

    Parameters
    ----------
    S0 : float
        Prix initial du sous-jacent.
    K : float
        Strike de l'option.
    T : float
        Maturité (en années).
    r : float
        Taux sans risque continu.
    sigma : float
        Volatilité (> 0).

    Returns
    -------
    float
        Prix du put européen.
    """
    return float(black_scholes_call(S0, K, T, r, sigma) - S0 + K * np.exp(-r * T))


def bs_vega(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
) -> float:
    """
    Vega d'un call (ou put) européen Black-Scholes.

        vega = S0 * sqrt(T) * phi(d1)

    où phi est la densité de la loi normale standard.

    Parameters
    ----------
    S0 : float
    K : float
    T : float
    r : float
    sigma : float

    Returns
    -------
    float
    """
    if T <= 0.0 or sigma <= 0.0:
        return 0.0
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    return float(S0 * sqrt_T * norm.pdf(d1))


# ─────────────────────────────────────────────────────────────────────────────
# Formule analytique de Merton
# ─────────────────────────────────────────────────────────────────────────────


def merton_call_analytic(
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
    Prix d'un call européen dans le modèle de Merton par la formule analytique exacte.

    Formule (Merton, 1976, eq. (15)) — dérivation par conditionnement sur N_T :

        C_Merton = S_0 * sum_{n=0}^{inf} w'_n * N(d1_n)
                 - K * e^{-rT} * sum_{n=0}^{inf} w_n * N(d2_n)

    avec :
        k_bar    = exp(mu_J + delta^2/2) - 1        (saut moyen en %)
        lambda'  = lambda * (1 + k_bar)              (intensité risque-neutre modifiée)
        w'_n     = e^{-lambda'*T} * (lambda'*T)^n / n!  (poids pour N(d1))
        w_n      = e^{-lambda*T}  * (lambda*T)^n  / n!  (poids pour N(d2))
        v_n      = sqrt(sigma^2*T + n*delta^2)       (vol. totale conditionnelle)
        d1_n     = [log(S0/K) + (r - lambda*k_bar)*T + n*(mu_J + delta^2/2)] / v_n
        d2_n     = d1_n - v_n

    La distinction entre lambda' (pour N(d1)) et lambda (pour N(d2)) découle de la
    structure non-symétrique de la formule de call : la composante S_0*N(d1) porte
    les (1+k_bar)^n facteurs des sauts, absorbés dans lambda', tandis que K*e^{-rT}*N(d2)
    est pondéré par la loi Poisson de paramètre original lambda.

    Parameters
    ----------
    S0 : float
        Prix initial du sous-jacent.
    K : float
        Strike de l'option.
    T : float
        Maturité (en années).
    r : float
        Taux sans risque continu.
    sigma : float
        Volatilité diffusive (> 0).
    lambda_ : float
        Intensité des sauts sous Q (> 0).
    mu_J : float
        Moyenne du log-saut : log(Y_i) ~ N(mu_J, delta^2).
    delta : float
        Écart-type du log-saut (> 0).
    n_terms : int, optional
        Nombre de termes dans la série. Défaut : 50.

    Returns
    -------
    float
        Prix du call européen dans le modèle de Merton.

    Notes
    -----
    La formule est dérivée en conditionnant sur N_T ~ Poisson(lambda*T).
    Conditionnellement à N_T = n, le log-prix est gaussien :
        log(S_T/S_0) | N_T=n ~ N((r - lambda*k_bar - sigma^2/2)*T + n*mu_J,
                                   sigma^2*T + n*delta^2)
    Le prix du call s'obtient par la formule lognormale standard sur chaque terme.
    La somme en lambda' (resp. lambda) pour N(d1) (resp. N(d2)) résulte de l'absorption
    des facteurs (1+k_bar)^n dans les poids de la première somme.
    Voir Merton (1976, eq. 15) et Cont & Tankov (2004, Prop. 11.1).
    """
    if T <= 0.0:
        return float(max(S0 - K, 0.0))

    k_bar = np.exp(mu_J + 0.5 * delta**2) - 1.0
    # Intensité risque-neutre modifiée (absorbe les facteurs (1+k_bar)^n)
    lambda_prime = lambda_ * (1.0 + k_bar)

    sum_N1 = 0.0  # Somme pondérée par lambda' pour N(d1)
    sum_N2 = 0.0  # Somme pondérée par lambda  pour N(d2)

    lambda_T        = lambda_  * T
    lambda_prime_T  = lambda_prime * T

    for n in range(n_terms):
        # ── Poids Poisson pour N(d1) : Poisson(lambda') ───────────────────
        lw_prime = -lambda_prime_T + n * np.log(lambda_prime_T + 1e-300) - _log_factorial(n)
        w_prime  = np.exp(lw_prime)

        # ── Poids Poisson pour N(d2) : Poisson(lambda) ────────────────────
        lw = -lambda_T + n * np.log(lambda_T + 1e-300) - _log_factorial(n)
        w  = np.exp(lw)

        # Critère d'arrêt : les deux poids sont négligeables
        if w_prime < 1e-15 and w < 1e-15:
            if n > max(lambda_prime_T, lambda_T) + 10:
                break

        # ── d1_n et d2_n exacts (formule lognormale conditionnelle) ───────
        # v_n^2 = Var[log(S_T/S_0) | N_T=n] = sigma^2*T + n*delta^2
        v_n2 = sigma**2 * T + n * delta**2
        v_n  = np.sqrt(v_n2) if v_n2 > 1e-20 else 1e-10

        # log(S0/K) + E[log(S_T/S_0) | N_T=n] + v_n^2/2
        # = log(S0/K) + (r - lambda*k_bar)*T + n*(mu_J + delta^2/2)
        log_FK = np.log(S0 / K) + (r - lambda_ * k_bar) * T + n * (mu_J + 0.5 * delta**2)

        d1_n = (log_FK + 0.5 * v_n2) / v_n
        d2_n = d1_n - v_n

        sum_N1 += w_prime * float(_norm_cdf(d1_n))
        sum_N2 += w        * float(_norm_cdf(d2_n))

    return float(S0 * sum_N1 - K * np.exp(-r * T) * sum_N2)


def _log_factorial(n: int) -> float:
    """log(n!) via l'approximation de Stirling pour n grand, exacte pour n petit."""
    if n <= 1:
        return 0.0
    return float(np.sum(np.log(np.arange(2, n + 1))))


# ─────────────────────────────────────────────────────────────────────────────
# Monte-Carlo Merton
# ─────────────────────────────────────────────────────────────────────────────


def merton_call_mc(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    lambda_: float,
    mu_J: float,
    delta: float,
    N_paths: int = 100_000,
    use_control_variate: bool = True,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Prix Monte-Carlo d'un call européen dans le modèle de Merton.

    Méthode : simulation directe du prix terminal S_T sous Q.

    Sous la mesure risque-neutre Q (Merton 1976), le prix terminal est :

        S_T = S_0 * exp((r - lambda*k_bar - sigma^2/2)*T
                        + sigma*sqrt(T)*Z
                        + sum_{i=1}^{N_T} log(Y_i))

    avec Z ~ N(0,1), N_T ~ Poisson(lambda*T), log(Y_i) ~ N(mu_J, delta^2).

    Variable de contrôle (optionnelle) : la composante brownienne seule,
    i.e. le payoff Black-Scholes sur un sous-jacent fictif sans sauts de variance
    sigma^2 * T (réduit la variance de 60-80% en pratique).

    Parameters
    ----------
    S0 : float
    K : float
    T : float
    r : float
    sigma : float
    lambda_ : float
    mu_J : float
    delta : float
    N_paths : int, optional
        Nombre de trajectoires Monte-Carlo. Défaut : 100 000.
    use_control_variate : bool, optional
        Si True, utilise la variable de contrôle Black-Scholes. Défaut : True.
    seed : int, optional
        Graine pour la reproductibilité. Défaut : 42.

    Returns
    -------
    price : float
        Estimation Monte-Carlo du prix du call.
    std_error : float
        Erreur standard de l'estimateur (intervalle de confiance à 95 % : ±1.96*std_error).

    Notes
    -----
    La variable de contrôle Black-Scholes est implémentée selon la méthode de
    réduction de variance standard. Le coefficient optimal beta* est estimé
    par régression OLS sur le même lot de simulations.
    Voir Glasserman (2003), Monte Carlo Methods in Financial Engineering, Ch. 4.
    """
    rng = np.random.default_rng(seed)
    k_bar = np.exp(mu_J + 0.5 * delta**2) - 1.0

    # ── Simulation du log-prix terminal sous Q ────────────────────────────
    # Composante brownienne
    Z = rng.standard_normal(N_paths)
    diffusion = (r - lambda_ * k_bar - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z

    # Composante de sauts : nombre de sauts par chemin
    N_jumps = rng.poisson(lambda_ * T, size=N_paths)
    max_jumps = int(N_jumps.max()) if N_jumps.max() > 0 else 0

    jump_sum = np.zeros(N_paths)
    if max_jumps > 0:
        log_Y = rng.normal(mu_J, delta, size=(N_paths, max_jumps))
        mask = np.arange(max_jumps)[np.newaxis, :] < N_jumps[:, np.newaxis]
        jump_sum = (log_Y * mask).sum(axis=1)

    # Prix terminaux
    S_T = S0 * np.exp(diffusion + jump_sum)

    # Payoffs actualises
    payoffs = np.exp(-r * T) * np.maximum(S_T - K, 0.0)

    if not use_control_variate:
        price = float(payoffs.mean())
        std_error = float(payoffs.std(ddof=1) / np.sqrt(N_paths))
        return price, std_error

    # Variable de controle Black-Scholes
    S_T_bs = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    payoffs_bs_raw = np.exp(-r * T) * np.maximum(S_T_bs - K, 0.0)

    bs_price_exact = black_scholes_call(S0, K, T, r, sigma)

    cov = np.cov(payoffs, payoffs_bs_raw)
    beta_star = cov[0, 1] / (cov[1, 1] + 1e-300)

    payoffs_cv = payoffs - beta_star * (payoffs_bs_raw - bs_price_exact)
    price = float(payoffs_cv.mean())
    std_error = float(payoffs_cv.std(ddof=1) / np.sqrt(N_paths))

    return price, std_error
