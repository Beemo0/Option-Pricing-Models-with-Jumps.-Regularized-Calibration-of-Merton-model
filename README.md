# Option Pricing Models with Jumps. Regularized Calibration of Merton model.

**PFE MMF ING3 — CY Tech 2025–2026**  
Corentin Stephan & Hugo Landron  
Supervisors: Y. Aktar · I. Kortchemski

---

## Abstract

An important issue in finance is model calibration. The calibration
problem is the inverse of the option pricing problem. It can be shown
that the usual formulation of the inverse problem via Non-linear Least Squares is an ill posed problem. 
To achieve well-posedness of the problem, some regularization is needed. This PFE consists of two parts.
The first one is the study of the Merton model, the simulation of the evolution of underlying asset with jumps. The analytical calculation
of the price of the options in the Merton model and its evaluation by Monte-Carlo method is required. 
The second part is the study a regularization method based on Lavenberg - Marquart regularisation and
on relative entropy introduced in the article of Rama Cont and Peter Tankov where the authors reformulate the calibration problem into a
problem of finding a risk-neutral exponential Levy model that reproduces the observed option prices and has the smallest possible relative entropy with respect to a chosen prior model.


## Overview

This repository implements the full pipeline of the Merton (1976) jump-diffusion model for European option pricing and regularized calibration on an implied volatility surface.

The project is divided into two parts:

- **Part 1** — Model study: simulation, analytical pricing, Monte-Carlo validation, and implied volatility smile analysis.
- **Part 2** — Inverse problem: Levenberg-Marquardt calibration with Tikhonov regularization and relative entropy regularization (Cont-Tankov, 2004).

---

## Repository Structure

```
├── Part 1/
│   ├── 01_simulation.ipynb          # Merton process simulation and stylized facts
│   ├── 02_pricing_validation.ipynb  # Analytical formula vs. Monte-Carlo validation
│   └── 03_smile_shape.ipynb         # Implied volatility smile analysis
│
├── Part 2/
│   ├── 04_calibration_LM.ipynb      # LM calibration and ill-posedness illustration
│   ├── 05_tikhonov.ipynb            # Tikhonov regularization and L-curve
│   └── 06_entropy_calibration.ipynb # Relative entropy regularization (Cont-Tankov)
│
└── src/
    └── merton_calib/
        ├── __init__.py
        ├── simulation.py            # Exact simulation of Merton trajectories
        ├── pricing.py               # Analytical Merton formula and Monte-Carlo pricer
        ├── implied_vol.py           # Black-Scholes implied volatility inversion (Brent)
        └── calibration.py           # LM algorithm, Tikhonov and entropy regularization
```

---

## Core Modules (`src/merton_calib/`)

### `simulation.py`
Exact simulation of the Merton jump-diffusion process under the historical measure $\mathbb{P}$. Implements the discrete exact scheme at each time step, combining the Brownian diffusion component with compound Poisson jumps of log-normal amplitude.

### `pricing.py`
Two pricing methods for European call options:
- **Analytical formula**: infinite series of Black-Scholes prices weighted by a Poisson distribution, truncated at $N = 20$ terms.
- **Monte-Carlo estimator**: direct simulation under the risk-neutral measure $\mathbb{Q}$, with Black-Scholes control variate for variance reduction.

### `implied_vol.py`
Numerical inversion of the Black-Scholes formula to compute implied volatility. Implemented in pure Python using the Brent root-finding algorithm, without external solvers.

### `calibration.py`
Full calibration pipeline for the reduced problem $(\lambda, \delta)$ with $\sigma$ and $\mu_J$ fixed:
- **Levenberg-Marquardt algorithm** with multi-start strategy ($n_{\text{start}} = 10$), finite-difference Jacobian ($h = 10^{-4}$), and adaptive damping parameter.
- **Tikhonov regularization**: augmented residual formulation, L-curve method for parameter selection.
- **Relative entropy regularization** (Cont-Tankov): analytical expression of the KL divergence for the Merton model, exact gradient computation.

---

## Notebooks

### Part 1 — Model Analysis

| Notebook | Description | Key outputs |
|---|---|---|
| `01_simulation.ipynb` | Simulates Merton trajectories and computes log-return distribution statistics | Trajectory plots, return histogram, excess kurtosis and skewness |
| `02_pricing_validation.ipynb` | Compares analytical formula against Monte-Carlo on a $4 \times 5$ option grid | Validation table (relative error < 1%), convergence curve |
| `03_smile_shape.ipynb` | Studies the implied volatility smile as a function of $\lambda$, $\delta$, $\mu_J$, and $T$ | Smile surface, skew quantification, term structure |

### Part 2 — Calibration

| Notebook | Description | Key outputs |
|---|---|---|
| `04_calibration_LM.ipynb` | Illustrates the ill-posedness of the calibration problem | Objective function contour map, multi-start trajectories, noise sensitivity |
| `05_tikhonov.ipynb` | Tikhonov regularization: L-curve, calibrated smile, temporal stability | L-curve, smile fit, 30-day parameter time series |
| `06_entropy_calibration.ipynb` | Relative entropy regularization: comparison with Tikhonov | Entropy map, L-curve, stability comparison table |

---

## Reference Parameters

All experiments use the following reference parameters:

| Parameter | Symbol | Value |
|---|---|---|
| Initial price | $S_0$ | 100 |
| Risk-free rate | $r$ | 5% |
| Diffusive volatility | $\sigma$ | 20% |
| Jump intensity | $\lambda$ | 1.0 jump/year |
| Mean log-jump | $\mu_J$ | −10% |
| Log-jump std dev | $\delta$ | 15% |

Fixed parameters for calibration: $\sigma_0 = 20\%$, $\mu_{J,0} = -10\%$.  
Prior model: $\theta_0 = (\lambda_0, \delta_0) = (0.8,\ 0.12)$.  
Target parameters: $(\lambda^*, \delta^*) = (1.0,\ 0.15)$.

---

## Requirements

This project is implemented in **pure NumPy** without external optimization libraries (no SciPy, no QuantLib).

```
Python >= 3.10
numpy
matplotlib
```

Install dependencies:

```bash
pip install numpy matplotlib
```

---

## Key Results

| Experiment | Result |
|---|---|
| E2 — Analytical vs. MC validation | Max relative error: 0.746% < 1% on 20 options |
| E3 — Implied volatility smile | Negative skew: 2.52 vol-points at $T = 1$ year |
| E4 — Ill-posedness | ±30% uncertainty on $\lambda^*$ for 0.5% measurement noise |
| E5 — Tikhonov regularization | 4× reduction in temporal variance of $\lambda^*$ |
| E6 — Relative entropy | 5× reduction in temporal variance; 25% better than Tikhonov |

---

## References

- Merton, R.C. (1976). *Option Pricing When Underlying Stock Returns Are Discontinuous*. Journal of Financial Economics, 3, 125–144.
- Cont, R. & Tankov, P. (2004). *Financial Modelling with Jump Processes*. Chapman & Hall/CRC.
- Cont, R. & Tankov, P. (2004). *Non-Parametric Calibration of Jump-Diffusion Option Pricing Models*. Journal of Computational Finance, 7(3), 1–49.
- He, C., Kennedy, J.S., Coleman, T., Forsyth, P.A., Li, Y., Vetzal, K. (2006). *Calibration and Hedging under Jump Diffusion*. Review of Derivatives Research, 9, 1–35.
- Tikhonov, A.N. & Arsenin, V.Y. (1977). *Solutions of Ill-Posed Problems*. Wiley.
- Hansen, P.C. (1992). *Analysis of Discrete Ill-Posed Problems by Means of the L-Curve*. SIAM Review, 34(4), 561–580.
