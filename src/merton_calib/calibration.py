# -*- coding: utf-8 -*-
"""
calibration.py — Calibration régularisée du modèle de Merton.

Implémente la calibration du problème réduit (λ, δ) avec σ et μ_J fixés,
par l'algorithme de Levenberg-Marquardt avec deux régularisations :

  1. Tikhonov       : L_α(λ,δ) = (1/N)‖F‖² + α‖θ − θ₀‖²
  2. Entropie (C-T) : L_β(λ,δ) = (1/N)‖F‖² + β · H(Q|Q₀)/T

où F_i = σ_imp^model(λ,δ,Kᵢ,Tᵢ) − σ_mkt_i.

References
----------
Cont, R. & Tankov, P. (2004). Non-parametric calibration of jump-diffusion
    option pricing models. Journal of Computational Finance, 7(3), 1-49.
He, C. et al. (2006). Calibration and hedging under jump diffusion.
    Review of Derivatives Research, 9(1), 1-35.
Levenberg, K. (1944). A method for the solution of certain non-linear
    problems in least squares. QJAM, 2(2), 164-168.
Marquardt, D.W. (1963). An algorithm for least-squares estimation of
    nonlinear parameters. J. SIAM, 11(2), 431-441.
"""

from __future__ import annotations

import math as _math
import numpy as np
from numpy.typing import NDArray

from .implied_vol import merton_implied_vol


# ─────────────────────────────────────────────────────────────────────────────
# Résidu et jacobienne
# ─────────────────────────────────────────────────────────────────────────────

def compute_residuals(
    lam: float,
    delta: float,
    sigma_mkt: NDArray[np.float64],
    strikes: NDArray[np.float64],
    maturities: NDArray[np.float64],
    S0: float,
    r: float,
    sigma0: float,
    muJ0: float,
    n_terms: int = 50,
) -> NDArray[np.float64]:
    """
    Vecteur résidu F ∈ R^N pour la calibration sur les volatilités implicites.

        F_i = σ_imp^model(λ, δ, Kᵢ, Tᵢ) − σ_mkt_i

    Le vecteur normalisé utilisé dans L = (1/N)‖F‖² est F/√N.

    Parameters
    ----------
    lam : float        Intensité λ > 0.
    delta : float      Écart-type du log-saut δ > 0.
    sigma_mkt : array  Volatilités implicites de marché (N,).
    strikes : array    Strikes Kᵢ (N,).
    maturities : array Maturités Tᵢ (N,).
    S0, r : float      Prix initial et taux sans risque.
    sigma0 : float     Volatilité diffusive fixée.
    muJ0 : float       Moyenne du log-saut fixée.
    n_terms : int      Termes dans la série de Merton.

    Returns
    -------
    F : ndarray (N,)   Résidus en points de volatilité.
    """
    N = len(sigma_mkt)
    F = np.empty(N)
    for i in range(N):
        sig_model = merton_implied_vol(
            S0, strikes[i], maturities[i], r,
            sigma0, lam, muJ0, delta, n_terms
        )
        if np.isnan(sig_model):
            F[i] = 10.0  # Pénalité forte si inversion échoue
        else:
            F[i] = sig_model - sigma_mkt[i]
    return F


def compute_jacobian(
    lam: float,
    delta: float,
    sigma_mkt: NDArray[np.float64],
    strikes: NDArray[np.float64],
    maturities: NDArray[np.float64],
    S0: float,
    r: float,
    sigma0: float,
    muJ0: float,
    h: float = 1e-4,
    n_terms: int = 50,
) -> NDArray[np.float64]:
    """
    Jacobienne J ∈ R^(N×2) par différences finies centrées (ordre O(h²)).

        J[:,0] = ∂F/∂λ ≈ [F(λ+h,δ) − F(λ−h,δ)] / (2h)
        J[:,1] = ∂F/∂δ ≈ [F(λ,δ+h) − F(λ,δ−h)] / (2h)

    Parameters
    ----------
    h : float   Pas de différentiation (défaut 1e-4).

    Returns
    -------
    J : ndarray (N, 2)
    """
    kw = dict(sigma_mkt=sigma_mkt, strikes=strikes, maturities=maturities,
              S0=S0, r=r, sigma0=sigma0, muJ0=muJ0, n_terms=n_terms)

    lam_p = max(lam + h, 1e-8)
    lam_m = max(lam - h, 1e-8)
    dF_dlam = (compute_residuals(lam_p, delta, **kw) -
               compute_residuals(lam_m, delta, **kw)) / (lam_p - lam_m)

    delta_p = max(delta + h, 1e-8)
    delta_m = max(delta - h, 1e-8)
    dF_ddelta = (compute_residuals(lam, delta_p, **kw) -
                 compute_residuals(lam, delta_m, **kw)) / (delta_p - delta_m)

    return np.column_stack([dF_dlam, dF_ddelta])


# ─────────────────────────────────────────────────────────────────────────────
# Entropie relative (Cont & Tankov 2004, eq. 5.38 de ce mémoire)
#
#  H(Q|Q₀)/T = λ·ln(λ/λ₀) + λ·ln(δ₀/δ) − (λ/2)(1 − δ²/δ₀²) − λ + λ₀
#
# Gradient :
#  ∂(H/T)/∂λ = ln(λ/λ₀) + ln(δ₀/δ) + (1/2)(δ²/δ₀² − 1)
#  ∂(H/T)/∂δ = λ(δ/δ₀² − 1/δ)
#
# Hessienne :
#  ∂²(H/T)/∂λ²  = 1/λ
#  ∂²(H/T)/∂λ∂δ = δ/δ₀² − 1/δ
#  ∂²(H/T)/∂δ²  = λ(1/δ₀² + 1/δ²)
# ─────────────────────────────────────────────────────────────────────────────

def entropy_KL(
    lam: float,
    delta: float,
    lam0: float,
    delta0: float,
) -> float:
    """
    Entropie relative H(Q|Q₀)/T dans le modèle de Merton (problème réduit).

    Formule analytique issue du calcul de la divergence KL entre deux processus
    de Poisson composés log-normaux de même moyenne de saut μ_J (fixé), mais
    d'intensités λ, λ₀ et d'écarts-types δ, δ₀ distincts.

    H(Q|Q₀)/T = λ·ln(λ/λ₀) + λ·ln(δ₀/δ) − (λ/2)(1 − δ²/δ₀²) − λ + λ₀

    Parameters
    ----------
    lam, delta   : Paramètres du modèle calibré Q.
    lam0, delta0 : Paramètres du modèle a priori Q₀.

    Returns
    -------
    float   Entropie relative (≥ 0, nulle si (λ,δ) = (λ₀,δ₀)).
    """
    if lam <= 0 or delta <= 0:
        return float('inf')
    return (lam * _math.log(lam / lam0)
            + lam * _math.log(delta0 / delta)
            - 0.5 * lam * (1.0 - delta**2 / delta0**2)
            - lam + lam0)


def entropy_grad(
    lam: float,
    delta: float,
    lam0: float,
    delta0: float,
) -> NDArray[np.float64]:
    """
    Gradient ∇(H/T) = [∂H/∂λ, ∂H/∂δ].

    ∂(H/T)/∂λ = ln(λ/λ₀) + ln(δ₀/δ) + (1/2)(δ²/δ₀² − 1)
    ∂(H/T)/∂δ = λ(δ/δ₀² − 1/δ)
    """
    dH_dlam = (_math.log(lam / lam0)
               + _math.log(delta0 / delta)
               + 0.5 * (delta**2 / delta0**2 - 1.0))
    dH_ddelta = lam * (delta / delta0**2 - 1.0 / delta)
    return np.array([dH_dlam, dH_ddelta])


def entropy_hess(
    lam: float,
    delta: float,
    lam0: float,
    delta0: float,
) -> NDArray[np.float64]:
    """
    Hessienne ∇²(H/T) ∈ R^(2×2).

    [[1/λ,              δ/δ₀² − 1/δ  ],
     [δ/δ₀² − 1/δ,   λ(1/δ₀² + 1/δ²)]]
    """
    off = delta / delta0**2 - 1.0 / delta
    H = np.array([
        [1.0 / lam,         off],
        [off,  lam * (1.0 / delta0**2 + 1.0 / delta**2)]
    ])
    return H


# ─────────────────────────────────────────────────────────────────────────────
# Algorithme de Levenberg-Marquardt
# ─────────────────────────────────────────────────────────────────────────────

def _lm_solve(A: NDArray, b: NDArray) -> NDArray:
    """Résout A x = b par décomposition de Cholesky (ou numpy.linalg.solve)."""
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.zeros_like(b)


def levenberg_marquardt(
    F_func,
    J_func,
    theta0: NDArray[np.float64],
    N_data: int,
    *,
    mu0: float = 1e-2,
    nu: float = 10.0,
    rho_min: float = 0.25,
    eps1: float = 1e-8,
    eps2: float = 1e-6,
    eps3: float = 1e-8,
    k_max: int = 500,
    bounds: tuple[float, float] = (1e-6, 50.0),
) -> dict:
    """
    Algorithme de Levenberg-Marquardt générique pour un problème de la forme :

        min_θ  (1/N)‖F(θ)‖² + R(θ)

    où R(θ) représente le terme de régularisation (Tikhonov ou entropie),
    déjà incorporé dans F_func/J_func sous forme augmentée (Tikhonov) ou via
    gradient/Hessienne (entropie).

    F_func(theta) → (F_tilde, J_tilde) pour Tikhonov (résidu augmenté)
    F_func(theta) → (grad_total, H_approx) pour entropie (gradient + Hessien GN)

    En pratique, on utilise l'interface unifiée :

        F_func(theta) → (res_vec, jac_mat)

    où :
      - res_vec ∈ R^(N+k) est le vecteur résidu augmenté (Tikhonov) ou un
        vecteur artificiel tel que ‖res_vec‖² = L(θ)
      - jac_mat ∈ R^((N+k)×2) est la jacobienne correspondante

    La mise à jour LM est :
        (J^T J + μ I) Δθ = −J^T F

    Parameters
    ----------
    F_func  : Callable  theta → (res, jac).
    J_func  : None      (inutilisé, inclus pour compatibilité).
    theta0  : ndarray (2,)  Point initial (λ₀, δ₀).
    N_data  : int       Nombre d'options (pour le calcul du RMSE).
    mu0     : float     Amortissement initial.
    nu      : float     Facteur d'adaptation de μ.
    rho_min : float     Seuil d'acceptation du pas (ρ_min ∈ (0,1)).
    eps1    : float     Tolérance gradient.
    eps2    : float     Tolérance pas relatif.
    eps3    : float     Tolérance variation de L.
    k_max   : int       Nombre maximal d'itérations.
    bounds  : tuple     Bornes (min, max) pour chaque paramètre.

    Returns
    -------
    dict avec clés : theta, lambda_, delta, L, RMSE, n_iter, converged, history
    """
    theta = np.clip(theta0.astype(float), bounds[0], bounds[1])
    mu = float(mu0)

    F, J = F_func(theta)
    L = float(np.dot(F, F))

    hist_L     = [L]
    hist_theta = [theta.copy()]
    hist_mu    = [mu]

    converged = False
    n_iter = 0

    for k in range(k_max):
        n_iter = k + 1
        JTF = J.T @ F          # (2,)
        JTJ = J.T @ J          # (2,2)

        grad_inf = float(np.max(np.abs(JTF)))

        # ── Critère 1 : gradient faible ───────────────────────────────────
        if grad_inf <= eps1:
            converged = True
            break

        # ── Résolution du système normal modifié ──────────────────────────
        A = JTJ + mu * np.eye(2)
        dtheta = _lm_solve(A, -JTF)

        # ── Critère 2 : pas relatif faible ────────────────────────────────
        step_rel = float(np.linalg.norm(dtheta) /
                         (np.linalg.norm(theta) + eps2))
        if step_rel <= eps2:
            converged = True
            break

        # ── Évaluation du ratio de gain ───────────────────────────────────
        theta_new = np.clip(theta + dtheta, bounds[0], bounds[1])
        F_new, J_new = F_func(theta_new)
        L_new = float(np.dot(F_new, F_new))

        # Prédiction linéaire : ‖F + J·Δθ‖²
        F_pred = F + J @ dtheta
        L_pred = float(np.dot(F_pred, F_pred))
        denom  = L - L_pred

        rho = (L - L_new) / (denom + 1e-300) if abs(denom) > 1e-300 else 0.0

        # ── Critère 3 : variation de L faible ─────────────────────────────
        if abs(L - L_new) / (L + eps3) <= eps3 and rho > 0:
            theta = theta_new
            F, J  = F_new, J_new
            L     = L_new
            converged = True
            break

        # ── Mise à jour θ et μ ────────────────────────────────────────────
        if rho > rho_min:
            theta = theta_new
            F, J  = F_new, J_new
            L     = L_new
            mu    = max(mu / nu, 1e-16)
        else:
            mu = min(mu * nu, 1e10)

        hist_L.append(L)
        hist_theta.append(theta.copy())
        hist_mu.append(mu)

    # RMSE sur les N_data premières composantes
    RMSE = float(np.sqrt(np.dot(F[:N_data], F[:N_data]) / N_data))

    return {
        'theta'     : theta,
        'lambda_'   : float(theta[0]),
        'delta'     : float(theta[1]),
        'L'         : L,
        'RMSE'      : RMSE,
        'n_iter'    : n_iter,
        'converged' : converged,
        'history'   : {
            'L'     : np.array(hist_L),
            'theta' : np.array(hist_theta),
            'mu'    : np.array(hist_mu),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Calibration avec régularisation de Tikhonov
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_tikhonov(
    sigma_mkt: NDArray[np.float64],
    strikes: NDArray[np.float64],
    maturities: NDArray[np.float64],
    S0: float,
    r: float,
    sigma0: float,
    muJ0: float,
    alpha: float,
    lam0: float,
    delta0: float,
    *,
    n_starts: int = 10,
    seed: int = 42,
    n_terms: int = 50,
    **lm_kwargs,
) -> dict:
    """
    Calibration par Levenberg-Marquardt avec régularisation de Tikhonov.

    Minimise la fonction augmentée :
        L_α(λ,δ) = (1/N)‖F‖² + α[(λ−λ₀)² + (δ−δ₀)²]

    via le résidu augmenté :
        F̃ = [F/√N ; √α(λ−λ₀) ; √α(δ−δ₀)] ∈ R^(N+2)

    Stratégie multi-départ : n_starts points initiaux tirés uniformément
    dans [0.05, 5] × [0.02, 0.5], meilleur résidu retenu.

    Parameters
    ----------
    sigma_mkt  : Volatilités implicites de marché (N,).
    strikes    : Strikes (N,).
    maturities : Maturités (N,).
    S0, r      : Prix initial, taux.
    sigma0     : Volatilité diffusive fixée.
    muJ0       : Moyenne du log-saut fixée.
    alpha      : Paramètre de régularisation Tikhonov (≥ 0).
    lam0, delta0 : Paramètre a priori θ₀.
    n_starts   : Nombre de points de départ.
    seed       : Graine aléatoire.
    n_terms    : Termes dans la série de Merton.

    Returns
    -------
    dict  Résultat de la meilleure calibration + 'all_starts' : liste de tous
          les résultats.
    """
    N = len(sigma_mkt)
    sqrt_N  = np.sqrt(N)
    sqrt_a  = np.sqrt(alpha)

    kw_res = dict(sigma_mkt=sigma_mkt, strikes=strikes, maturities=maturities,
                  S0=S0, r=r, sigma0=sigma0, muJ0=muJ0, n_terms=n_terms)

    def F_aug(theta):
        lam, delta = float(theta[0]), float(theta[1])
        F = compute_residuals(lam, delta, **kw_res)
        J = compute_jacobian(lam, delta, **kw_res)
        # Résidu augmenté
        F_tilde = np.concatenate([F / sqrt_N,
                                   [sqrt_a * (lam   - lam0),
                                    sqrt_a * (delta - delta0)]])
        # Jacobienne augmentée
        J_aug_row_lam   = np.array([sqrt_a, 0.0])
        J_aug_row_delta = np.array([0.0, sqrt_a])
        J_tilde = np.vstack([J / sqrt_N,
                              J_aug_row_lam[np.newaxis, :],
                              J_aug_row_delta[np.newaxis, :]])
        return F_tilde, J_tilde

    rng = np.random.default_rng(seed)
    starts = np.column_stack([
        rng.uniform(0.05, 5.0, n_starts),    # λ
        rng.uniform(0.02, 0.5, n_starts),    # δ
    ])
    # Toujours inclure θ₀ comme premier départ
    starts[0] = [lam0, delta0]

    all_results = []
    best = None

    for s in starts:
        res = levenberg_marquardt(
            F_aug, None, s, N,
            **lm_kwargs
        )
        all_results.append(res)
        if best is None or res['L'] < best['L']:
            best = res

    best = dict(best)
    best['all_starts'] = all_results
    best['alpha'] = alpha
    best['lam0']  = lam0
    best['delta0'] = delta0
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Calibration avec régularisation par entropie relative
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_entropy(
    sigma_mkt: NDArray[np.float64],
    strikes: NDArray[np.float64],
    maturities: NDArray[np.float64],
    S0: float,
    r: float,
    sigma0: float,
    muJ0: float,
    beta: float,
    lam0: float,
    delta0: float,
    *,
    n_starts: int = 10,
    seed: int = 42,
    n_terms: int = 50,
    **lm_kwargs,
) -> dict:
    """
    Calibration par Levenberg-Marquardt avec régularisation par entropie relative.

    Minimise :
        L_β(λ,δ) = (1/N)‖F‖² + β · H(Q(λ,δ)|Q₀)/T

    Formulation LM modifiée :
        (J^T J / N + β · H_H + μ I) Δθ = −(J^T F / N + β · g_H)

    où g_H = ∇(H/T) et H_H = ∇²(H/T) sont calculés analytiquement.
    On construit un "résidu pseudo-augmenté" en factorisant le terme
    d'entropie via la racine carrée de sa hessienne (Cholesky).

    Parameters
    ----------
    beta       : Paramètre de régularisation entropie (≥ 0).
    lam0, delta0 : Paramètre a priori Q₀.
    (autres)   : cf. calibrate_tikhonov.

    Returns
    -------
    dict  Résultat + 'all_starts'.
    """
    N = len(sigma_mkt)
    sqrt_N = np.sqrt(N)

    kw_res = dict(sigma_mkt=sigma_mkt, strikes=strikes, maturities=maturities,
                  S0=S0, r=r, sigma0=sigma0, muJ0=muJ0, n_terms=n_terms)

    def F_aug(theta):
        lam, delta = float(theta[0]), float(theta[1])
        F = compute_residuals(lam, delta, **kw_res)
        J = compute_jacobian(lam, delta, **kw_res)

        # Termes data (normalisés par N)
        JTJ_data = J.T @ J / N   # (2,2)
        JTF_data = J.T @ F / N   # (2,)

        # Termes entropie (analytiques)
        g_H   = entropy_grad(lam, delta, lam0, delta0) if beta > 0 else np.zeros(2)
        H_H   = entropy_hess(lam, delta, lam0, delta0) if beta > 0 else np.zeros((2, 2))
        H_ent = entropy_KL(lam, delta, lam0, delta0)   if beta > 0 else 0.0

        # Hessienne totale approx (Gauss-Newton data + exacte entropie)
        JTJ_total = JTJ_data + beta * H_H  # (2,2)

        # Construction d'un résidu et jacobienne artificiels tels que :
        #   J_art^T J_art = JTJ_total
        #   J_art^T F_art = JTF_data + beta * g_H
        # On utilise la factorisation de Cholesky de JTJ_total.
        try:
            L_chol = np.linalg.cholesky(JTJ_total + 1e-12 * np.eye(2))
        except np.linalg.LinAlgError:
            L_chol = np.eye(2) * 1e-6

        # F_art = L_chol^{-T} (JTF_data + beta * g_H)
        rhs = JTF_data + beta * g_H
        F_art = np.linalg.solve(L_chol.T, rhs)      # (2,)
        J_art = L_chol.T                              # (2,2)

        # L(θ) = (1/N)‖F‖² + β H/T
        L_val = np.dot(F, F) / N + beta * H_ent

        # Retourner un vecteur cohérent : ‖F_art‖² ≈ L(θ), J_art^T F_art = gradient/2
        return F_art, J_art

    rng = np.random.default_rng(seed)
    starts = np.column_stack([
        rng.uniform(0.05, 5.0, n_starts),
        rng.uniform(0.02, 0.5, n_starts),
    ])
    starts[0] = [lam0, delta0]

    all_results = []
    best = None

    for s in starts:
        res = levenberg_marquardt(
            F_aug, None, s, N,
            **lm_kwargs
        )
        # Recalculer L et RMSE avec la vraie fonction objectif
        lam_s, delta_s = res['lambda_'], res['delta']
        F_true = compute_residuals(lam_s, delta_s, **kw_res)
        H_val  = entropy_KL(lam_s, delta_s, lam0, delta0) if beta > 0 else 0.0
        res['L']    = float(np.dot(F_true, F_true) / N + beta * H_val)
        res['RMSE'] = float(np.sqrt(np.dot(F_true, F_true) / N))
        all_results.append(res)
        if best is None or res['L'] < best['L']:
            best = res

    best = dict(best)
    best['all_starts'] = all_results
    best['beta']   = beta
    best['lam0']   = lam0
    best['delta0'] = delta0
    return best


# ─────────────────────────────────────────────────────────────────────────────
# L-curve
# ─────────────────────────────────────────────────────────────────────────────

def lcurve(
    sigma_mkt: NDArray[np.float64],
    strikes: NDArray[np.float64],
    maturities: NDArray[np.float64],
    S0: float,
    r: float,
    sigma0: float,
    muJ0: float,
    lam0: float,
    delta0: float,
    *,
    method: str = 'tikhonov',
    reg_params: NDArray[np.float64] | None = None,
    n_starts: int = 5,
    seed: int = 42,
    n_terms: int = 50,
) -> dict:
    """
    Calcule la L-curve pour le choix du paramètre de régularisation.

    Trace le graphe logarithmique :
      (α ou β) → (log RMSE_data, log R(θ*))

    où R(θ*) est la norme de régularisation :
      - Tikhonov : ‖θ* − θ₀‖² = (λ*−λ₀)² + (δ*−δ₀)²
      - Entropie : H(Q(λ*,δ*)|Q₀)/T

    Parameters
    ----------
    method      : 'tikhonov' ou 'entropy'.
    reg_params  : Grille des paramètres de régularisation. Si None,
                  utilise une grille logarithmique sur [1e-6, 1e1].
    n_starts    : Points de départ pour chaque calibration (réduit pour rapidité).

    Returns
    -------
    dict avec clés :
      'reg_params' : grille des paramètres
      'RMSE'       : RMSE de calibration pour chaque α/β
      'reg_norm'   : norme de régularisation pour chaque α/β
      'theta_star' : paramètres calibrés (N_reg, 2)
      'L_total'    : valeur totale de la fonction objectif
    """
    if reg_params is None:
        reg_params = np.logspace(-6, 1, 20)

    kw_calib = dict(
        sigma_mkt=sigma_mkt, strikes=strikes, maturities=maturities,
        S0=S0, r=r, sigma0=sigma0, muJ0=muJ0,
        lam0=lam0, delta0=delta0, n_starts=n_starts, seed=seed, n_terms=n_terms
    )

    RMSEs     = []
    reg_norms = []
    thetas    = []
    L_totals  = []

    for reg in reg_params:
        if method == 'tikhonov':
            res = calibrate_tikhonov(alpha=float(reg), **kw_calib)
            lam_s, delta_s = res['lambda_'], res['delta']
            reg_n = (lam_s - lam0)**2 + (delta_s - delta0)**2
        elif method == 'entropy':
            res = calibrate_entropy(beta=float(reg), **kw_calib)
            lam_s, delta_s = res['lambda_'], res['delta']
            reg_n = entropy_KL(lam_s, delta_s, lam0, delta0)
        else:
            raise ValueError(f"method doit être 'tikhonov' ou 'entropy', reçu: {method!r}")

        RMSEs.append(res['RMSE'])
        reg_norms.append(reg_n)
        thetas.append([lam_s, delta_s])
        L_totals.append(res['L'])

    return {
        'reg_params' : reg_params,
        'RMSE'       : np.array(RMSEs),
        'reg_norm'   : np.array(reg_norms),
        'theta_star' : np.array(thetas),
        'L_total'    : np.array(L_totals),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Fonction utilitaire : surface synthétique
# ─────────────────────────────────────────────────────────────────────────────

def synthetic_smile(
    S0: float,
    r: float,
    sigma0: float,
    muJ0: float,
    lam_true: float,
    delta_true: float,
    strikes: NDArray[np.float64],
    maturities: NDArray[np.float64],
    noise_std: float = 0.0,
    seed: int = 42,
    n_terms: int = 50,
) -> NDArray[np.float64]:
    """
    Génère une surface de volatilité implicite synthétique.

    σ_mkt = σ_imp^model(λ_true, δ_true, K, T) + ε,   ε ~ N(0, noise_std²)

    Parameters
    ----------
    noise_std : float  Écart-type du bruit (en points de vol). 0 = sans bruit.

    Returns
    -------
    sigma_mkt : ndarray (N,)  Volatilités implicites (bruitées si noise_std > 0).
    """
    N = len(strikes)
    sigma_mkt = np.empty(N)
    for i in range(N):
        sigma_mkt[i] = merton_implied_vol(
            S0, strikes[i], maturities[i], r,
            sigma0, lam_true, muJ0, delta_true, n_terms
        )

    if noise_std > 0:
        rng = np.random.default_rng(seed)
        sigma_mkt += rng.normal(0.0, noise_std, N)

    return sigma_mkt
