"""
simulation.py — Simulation de trajectoires du modèle de Merton.

Schéma exact : entre deux sauts, le processus suit un mouvement brownien géométrique.
Les instants et amplitudes de sauts sont simulés directement à partir des lois de Poisson
et log-normale.

References
----------
Merton, R.C. (1976). Option pricing when underlying stock returns are discontinuous.
    Journal of Financial Economics, 3(1-2), 125-144.
Cont, R. & Tankov, P. (2004). Financial Modelling with Jump Processes.
    CRC Press. (Ch. 14)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def simulate_merton_paths(
    S0: float,
    mu: float,
    sigma: float,
    lambda_: float,
    mu_J: float,
    delta: float,
    T: float,
    N_steps: int,
    N_paths: int,
    seed: int = 42,
) -> NDArray[np.float64]:
    """
    Simule N_paths trajectoires discrètes du modèle de Merton sous la mesure réelle P.

    La dynamique du sous-jacent est (Merton 1976, eq. (7)) :

        dS_t / S_{t-} = (mu - lambda * k_bar) dt + sigma dW_t + d(sum_{i=1}^{N_t} (Y_i - 1))

    où k_bar = E[Y - 1] = exp(mu_J + delta^2/2) - 1.

    Le schéma utilisé est le schéma exact log-normal sur chaque pas dt :

        S_{t+dt} = S_t * exp((mu - lambda*k_bar - sigma^2/2)*dt
                              + sigma*sqrt(dt)*Z
                              + sum_{i=1}^{N_dt} log(Y_i))

    avec Z ~ N(0,1) et N_dt ~ Poisson(lambda*dt).

    Parameters
    ----------
    S0 : float
        Prix initial du sous-jacent.
    mu : float
        Rendement instantané espéré sous P (dérive réelle).
    sigma : float
        Volatilité diffusive (> 0).
    lambda_ : float
        Intensité du processus de Poisson (nombre moyen de sauts par unité de temps).
    mu_J : float
        Moyenne du log-saut : log(Y_i) ~ N(mu_J, delta^2).
    delta : float
        Écart-type du log-saut (> 0).
    T : float
        Horizon de simulation (en années).
    N_steps : int
        Nombre de pas de temps.
    N_paths : int
        Nombre de trajectoires à simuler.
    seed : int, optional
        Graine du générateur aléatoire pour la reproductibilité. Défaut : 42.

    Returns
    -------
    paths : ndarray of shape (N_paths, N_steps + 1)
        Tableau des prix simulés. paths[i, j] = S_{j*dt} pour la trajectoire i.

    Notes
    -----
    Le compensateur -lambda*k_bar dans la dérive garantit que E[dS_t/S_{t-}] = mu dt,
    conformément à la propriété de martingale sous la mesure réelle. Voir Merton (1976)
    et Cont & Tankov (2004, Ch. 14).
    """
    rng = np.random.default_rng(seed)
    dt = T / N_steps

    # Saut moyen en pourcentage (compensateur)
    k_bar = np.exp(mu_J + 0.5 * delta**2) - 1

    # Dérive ajustée (constante sur chaque pas)
    drift = (mu - lambda_ * k_bar - 0.5 * sigma**2) * dt

    paths = np.empty((N_paths, N_steps + 1), dtype=np.float64)
    paths[:, 0] = S0

    for step in range(N_steps):
        # ── Composante brownienne ──────────────────────────────────────────
        Z = rng.standard_normal(N_paths)
        diffusion = sigma * np.sqrt(dt) * Z

        # ── Composante de sauts ───────────────────────────────────────────
        # Nombre de sauts par trajectoire dans [t, t+dt]
        N_jumps = rng.poisson(lambda_ * dt, size=N_paths)

        # Somme des log-sauts : vectorisation par lot
        total_jumps = np.zeros(N_paths, dtype=np.float64)
        max_jumps = int(N_jumps.max()) if N_jumps.max() > 0 else 0
        if max_jumps > 0:
            # Tirage d'un bloc (N_paths x max_jumps) de log-sauts
            log_Y = rng.normal(mu_J, delta, size=(N_paths, max_jumps))
            # Masque : ne sommer que les sauts effectivement réalisés
            mask = np.arange(max_jumps)[np.newaxis, :] < N_jumps[:, np.newaxis]
            total_jumps = (log_Y * mask).sum(axis=1)

        # ── Mise à jour des prix ──────────────────────────────────────────
        paths[:, step + 1] = paths[:, step] * np.exp(drift + diffusion + total_jumps)

    return paths


def log_returns(paths: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Calcule les log-rendements sur chaque pas à partir d'un tableau de trajectoires.

    Parameters
    ----------
    paths : ndarray of shape (N_paths, N_steps + 1)
        Trajectoires de prix simulées.

    Returns
    -------
    returns : ndarray of shape (N_paths, N_steps)
        Log-rendements r_{t} = log(S_{t+1} / S_t).
    """
    return np.log(paths[:, 1:] / paths[:, :-1])
