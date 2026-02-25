############################################################################
### QPMwP CODING EXAMPLES - BLACK LITTERMAN MODEL - COUNTRY INDEXES DATASETS
############################################################################

# --------------------------------------------------------------------------
# Cyril Bachelard
# This version:     17.04.2025
# First version:    17.04.2025
# --------------------------------------------------------------------------



# This script demonstrates the application of the Black-Litterman model using 
# msci country index data.

# We will:
# 1. Load country index data
# 2. Compute the implied expected returns from market capitalization weights
# 3. Formulate investor views
# 4. Compute the posterior expected returns using the Black-Litterman model
# 5. Perform portfolio optimization using the posterior expected returns





# Standard library imports
import os
import sys

# Third party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add the project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
sys.path.append(project_root)
sys.path.append(src_path)

# Local modules imports
from btlight.helper_functions import load_data_msci
from estimation.covariance import Covariance
from estimation.black_litterman import bl_posterior_mu_sigma






# --------------------------------------------------------------------------
# Helper functions
# --------------------------------------------------------------------------

def mean_variance_weights(cov, mu, risk_aversion: float, budget=True):
    """
    Closed-form mean-variance optimal weights. If budget is True, the weights sum to one.

    Parameters
    ----------
    cov : (n, n) ndarray
        Covariance matrix
    mu : (n,) ndarray
        Expected returns
    risk_aversion : float
        Risk aversion parameter (gamma)
    budget : bool, optional
        If True, enforce the budget constraint that weights sum to one

    Returns
    -------
    w : (n,) ndarray
        Optimal portfolio weights
    """

    mu = np.asarray(mu)
    inv_cov = np.linalg.inv(cov)

    if not budget:
        w = (1 / risk_aversion) * inv_cov @ mu

    else:

        n = len(mu)
        ones = np.ones(n)


        lambda_ = (
            1 - (1 / risk_aversion) * (ones @ inv_cov @ mu)
        ) / (ones @ inv_cov @ ones)

        w = (1 / risk_aversion) * inv_cov @ mu + lambda_ * inv_cov @ ones

    return w





# --------------------------------------------------------------------------
# Load data
# --------------------------------------------------------------------------

N = 24
data = load_data_msci(path = '../data/', n = N)





# --------------------------------------------------------------------------
# Generate Cap-Weights
# See: https://www.msci.com/documents/10199/178e6643-6ae6-47b9-82be-e1fc565ededb
# --------------------------------------------------------------------------

country_names = [
    'US',
    'JP',
    'GB',
    'CA',
    'FR',
]
cap_weights = pd.Series(
    data=[
        0.7251,    # US
        # 0.0000,
        0.0547,    # JP
        0.0358,    # GB
        0.0334,    # CA
        # 0.0000,
        0.0261,    # FR
    ],
    index=country_names,
)
# Normalize such that weights sum to one
cap_weights = cap_weights / cap_weights.sum()
cap_weights






# --------------------------------------------------------------------------
# Implied Expected Returns (Prior)
# --------------------------------------------------------------------------

# Step 1: Compute the covariance matrix for the selected country indices
covariance = Covariance(method='pearson')
return_series = data['return_series'][country_names]
covmat = covariance.estimate(return_series, inplace=False) * 252  # Annualize the covariance matrix assuming 252 trading days
covmat


# Step 2. Calculate implied expected return from the cap-weights, covariance matrix and risk aversion coefficient
risk_aversion = 1
mu_implied = risk_aversion * covmat @ cap_weights
mu_implied

mu_implied.plot(kind='bar', figsize=(10, 5))

# Step 3: Assert that analytical solution to the unconstrained mean-variance optimization
# using the implied expected returns gives the same weights as the cap-weights
w_star = pd.Series(np.linalg.inv(covmat) @ mu_implied, index=country_names)

W = pd.DataFrame([cap_weights, w_star], index=["w_prior", "w_star"]).T
W.plot(kind="bar", figsize=(10, 5))
W
np.abs(W['w_star'] - W['w_prior']).max()

# Step 4: Show input sensitivity of the mean-variance optimization.
# For that, modify the implied expected returns and recompute the optimal weights.
mu_implied_mod = mu_implied.copy()
mu_implied_mod.iloc[0] = mu_implied_mod.iloc[0] * 0.8
mu_implied_mod = mu_implied_mod / mu_implied_mod.sum() * mu_implied.sum()
Mu = pd.concat({
    'mu_implied': mu_implied,
    'mu_implied_mod': mu_implied_mod,
}, axis=1)

Mu.plot(kind='bar', figsize=(10, 5))

# Optimal weights with modified expected returns
w_star_mod = pd.Series(np.linalg.inv(covmat) @ mu_implied_mod, index=country_names)
W = pd.DataFrame([cap_weights, w_star, w_star_mod], index=["w_prior", "w_star", "w_star_mod"]).T

W.plot(kind="bar", figsize=(10, 5))

Mu.corr()
W.corr()
return_series.corr()





# --------------------------------------------------------------------------
# Creating Views
# --------------------------------------------------------------------------

# Let's assume we have the following views:
# View 1: JP will have an expected return of 4% (vs. 0.8431% implied)
# View 2: The US will underperform the average of the other countries by 2%
# View 3: The US will outperform Japan by 1% (Notice that this somewhat contradicts Views 1 and 2)

P = pd.DataFrame(
    data=[
        [0, 1, 0, 0, 0],                    # Absolute view on JP
        [1, -0.25, -0.25, -0.25, -0.25],    # Relative view on US vs. others
        [1, -1, 0, 0, 0],                   # Relative view on US vs. JP
        # [0, 1, 0, 0],                    # Absolute view on JP
        # [1, -0.333, -0.333, -0.333],     # Relative view on US vs. others
        # [1, -1, 0, 0],                   # Relative view on US vs. JP
    ],
    index=['View1', 'View2', 'View3'],
    columns=country_names,
)

q = pd.Series(
    data=[0.04, -0.02, 0.01],  # Expected returns for each view - annualized
    index=P.index,
)

P
q



# --------------------------------------------------------------------------
# Tuning parameters
tau_psi = 0.05
tau_omega = 0.001
# --------------------------------------------------------------------------

# Uncertainty of the prior
Psi = covmat * tau_psi

# Uncertainty of the views
Omega = pd.DataFrame(
    np.diag([tau_omega] * len(q)),
    index=q.index,
    columns=q.index
)
Omega


# # Alternatively:
# Omega = P @ covmat @ P.T * 100
# Psi = covmat * 0.01




# --------------------------------------------------------------------------
# Posterior Expected Returns
# --------------------------------------------------------------------------

# Compute the posterior expected return vector
mu_posterior, sigma_posterior = bl_posterior_mu_sigma(
    mu_prior=mu_implied,
    covmat=covmat,
    P=P,
    q=q,
    Psi=Psi,
    Omega=Omega,
)

Mu = pd.concat({
    "mu_prior": mu_implied,
    "mu_posterior": mu_posterior,
}, axis=1)
Mu

Mu.plot(kind="bar", figsize=(10, 5))
Mu.corr()





# --------------------------------------------------------------------------
# Portfolio Optimization
# --------------------------------------------------------------------------

# Analytical mean-variance optimization with the posterior returns
w_star_post = pd.Series(mean_variance_weights(covmat, mu_posterior, risk_aversion, budget=False), index=country_names)
W = pd.DataFrame([cap_weights, w_star_post], index=["w_prior", "w_posterior"]).T
W

W.plot(kind="bar", figsize=(10, 5))
W.corr()



# Re-run the mean-variance optimization, using a solver, with the posterior returns
from optimization.optimization import MeanVariance, Objective
from optimization.constraints import Constraints


constraints = Constraints(country_names)
constraints.add_budget()
constraints.add_box(lower=0, upper=0.3)

mv = MeanVariance(
    constraints=constraints,
    solver_name="cvxopt",
)
mv.objective = Objective(
    q=mu_posterior * (-1),
    P=sigma_posterior,
)
mv.solve()
w_posterior = pd.Series(mv.results["weights"])


W = pd.DataFrame([cap_weights, w_star_post, w_posterior],
                 index=["w_prior", "w_posterior_analytical", "w_posterior_numerical"]).T
W.plot(kind="bar", figsize=(10, 5))
W.sum()


data['return_series'][country_names].corr()   # Notice the correlation between US and CA





# --------------------------------------------------------------------------
# Step-by-step - add each view independently and observe the effects to:
# - the posterior expected returns
# - the unconstrained analytical optimal portfolio weights
# - the budget-constrained analytical optimal portfolio weights
# - the long-only numerical optimal portfolio weights
# --------------------------------------------------------------------------

from optimization.optimization import MeanVariance, Objective
from optimization.constraints import Constraints

constraints = Constraints(country_names)
constraints.add_budget()
constraints.add_box(lower=0, upper=1.0)

mv = MeanVariance(
    constraints=constraints,
    solver_name="cvxopt",
)
mv.objective = Objective(
    q=mu_implied * (-1),
    P=covmat,
)

# Prep output
mu_post_dict = {'prior': mu_implied}
w_post_dict = {'prior': cap_weights}
w_post_budget_dict = {'prior': cap_weights}
w_post_lo_dict = {'prior': cap_weights}

for i in range(len(q)+2):

    if i <= (len(q) - 1):

        key = P.index[i]

        P_i = P.iloc[[i]]
        q_i = q.iloc[[i]]

        mu_post_i, sigma_post_i = bl_posterior_mu_sigma(
            mu_prior=mu_implied,
            covmat=covmat,
            P=P_i,
            q=q_i,
            Psi=Psi,
            Omega=Omega.iloc[[i], [i]],
        )
    elif i == len(q):
        key = 'linear_comb'
        mu_post_i = pd.concat(mu_post_dict, axis=1).mean(axis=1)
    else:
        key = 'all_views'
        mu_post_i, sigma_post_i = bl_posterior_mu_sigma(
            mu_prior=mu_implied,
            covmat=covmat,
            P=P,
            q=q,
            Psi=Psi,
            Omega=Omega,
        )

    mu_post_dict[key] = mu_post_i

    w_post_dict[key] = mean_variance_weights(
        cov=sigma_post_i,
        mu=mu_post_i,
        risk_aversion=risk_aversion,
        budget=False,
    )
    w_post_budget_dict[key] = mean_variance_weights(
        cov=sigma_post_i,
        mu=mu_post_i,
        risk_aversion=risk_aversion,
        budget=True,
    )
    mv.objective.coefficients['q'] = mu_post_i * (-1)
    mv.solve()
    w_post_lo_dict[key] = pd.Series(mv.results["weights"])



Mu_post = pd.DataFrame(mu_post_dict)
Mu_post.plot(kind="bar", figsize=(10, 5), title="Posterior Expected Returns")

W_post = pd.DataFrame(w_post_dict)
W_post.sum(axis=0)
W_post.plot(kind="bar", figsize=(10, 5), title="Unconstrained Optimal Weights")

W_post_budget = pd.DataFrame(w_post_budget_dict)
W_post_budget.sum(axis=0)
W_post_budget.plot(kind="bar", figsize=(10, 5), title="Budget-Constrained Optimal Weights")

W_post_lo = pd.DataFrame(w_post_lo_dict)
W_post_lo.sum(axis=0)
W_post_lo.plot(kind="bar", figsize=(10, 5), title="Long-Only Optimal Weights")
