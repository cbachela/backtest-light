############################################################################
### QPMwP - BLACK LITTERMAN
############################################################################

# --------------------------------------------------------------------------
# Cyril Bachelard
# This version:     28.04.2025
# First version:    28.04.2025
# --------------------------------------------------------------------------




# Standard library imports
from typing import Union

# Third party imports
import numpy as np
import pandas as pd






def bl_posterior_mu_sigma(
    mu_prior: pd.Series,
    covmat: pd.DataFrame, 
    P: Union[np.ndarray, pd.DataFrame],
    q: Union[np.ndarray, pd.Series],
    Psi: Union[np.ndarray, pd.DataFrame],
    Omega: Union[np.ndarray, pd.DataFrame],
    confidence: float = 1,
) -> pd.Series:
    """
    Computes the posterior mean vector and posterior covariance matrix under the
    Black–Litterman model using the Bayesian update:

    The investor holds a prior on expected returns,
    \

    \[
        \mu \sim \mathcal{N}(\mu_{\text{prior}},\, \Psi),
    \\]


    and expresses views of the form
    \

    \[
        q = P \mu + \varepsilon,\qquad 
        \varepsilon \sim \mathcal{N}(0,\, \Omega).
    \\]



    The posterior distribution of \\( \mu \\) is Gaussian with:

    **Posterior precision matrix**
    \

    \[
        V^{-1} = \Psi^{-1} + P^{\\top}\,\Omega^{-1}\,P.
    \\]



    **Posterior mean**
    \

    \[
        \mu_{\text{post}}
        = V \left( \Psi^{-1}\mu_{\text{prior}}
        + P^{\\top}\Omega^{-1} q \right).
    \\]



    **Posterior covariance of the mean estimate**
    \

    \[
        \Sigma_{\mu|\text{data}} = V
        = \left( \Psi^{-1} + P^{\\top}\Omega^{-1}P \right)^{-1}.
    \\]



    Note:
    \\( \Sigma_{\mu|\text{data}} \\) represents *uncertainty in the mean estimates*,
    not the covariance of asset returns.  
    A common heuristic (used here) is to adjust the prior covariance as:
    \

    \[
        \Sigma_{\text{post}} = \Sigma_{\text{prior}} + \Sigma_{\mu|\text{data}}.
    \\]



    Parameters
    ----------
    mu_prior : pd.Series
        Prior expected returns.
    covmat : pd.DataFrame
        Prior covariance matrix \\( \Sigma_{\text{prior}} \\).
    P : array-like
        Pick matrix encoding the views.
    q : array-like
        Expected returns for the views.
    Psi : array-like
        Prior uncertainty matrix (often \\( \tau \Sigma \\)).
    Omega : array-like
        View uncertainty matrix.
    confidence : float
        Scalar that rescales \\( \Omega \\) as \\( \Omega / \text{confidence} \\).

    Returns
    -------
    pd.Series
        Posterior expected returns \\( \mu_{\text{post}} \\).
    pd.DataFrame
        Posterior covariance matrix \\( \Sigma_{\text{post}} \\).
    """


    # Ensure all matrices have the same ordering
    # before converting them to numpy arrays
    ids = mu_prior.index
    if isinstance(mu_prior, pd.Series):
        mu_prior = mu_prior.to_numpy()
    if isinstance(P, pd.DataFrame):
        P = P[ids].to_numpy()
    if isinstance(q, pd.Series):
        q = q.to_numpy()
    if isinstance(Psi, pd.DataFrame):
        Psi = Psi.loc[ids, ids].to_numpy()
    if isinstance(Omega, pd.DataFrame):
        Omega = Omega.to_numpy()

    # Scale the uncertainty matrix of the views by 1/confidence
    Omega = Omega / confidence

    # ~~~~~~~~~~~~~~~~~~~~ TODO: review, use alternative formula
    # Compute the posterior mean and covariance
    Psi_inv = np.linalg.inv(Psi)
    Omega_inv = np.linalg.inv(Omega)
    V = P.T @ Omega_inv @ P + Psi_inv
    V_inv = np.linalg.inv(V)  # //Beware: this reflects uncertainty in the mean estimates, not the variability of returns.

    mu_posterior = V_inv @ (
        P.T @ Omega_inv @ q + Psi_inv @ mu_prior
    )
    sigma_posterior = covmat + pd.DataFrame(V_inv, index=ids, columns=ids)
    # ~~~~~~~~~~~~~~~~~~~~

    # # Alternative formula (computationally more stable, according to Meucci (2010))
    # mu_posterior = mu_prior + (tau * Sigma @ P.T) @ np.linalg.inv(tau * P @ Sigma @ P.T + Omega) @ (q - P @ mu_prior)
    # sigma_posterior = (1 - tau) * Sigma - tau**2 * Sigma @ P.T np.linalg.inv(tau * P @ Sigma @ P.T + Omega) @ (P @ Sigma)

    # Convert to pandas
    mu_posterior = pd.Series(mu_posterior, index=ids)
    # sigma_posterior = pd.DataFrame(sigma_posterior, index=ids, columns=ids)

    return mu_posterior, sigma_posterior


def view_from_scores_quintile(
    scores: pd.Series,
    mu_implied: pd.Series,
    scalefactor: int = 1,
) -> (pd.DataFrame, pd.Series):
    """
    Generate views based on quintile thresholds of scores.

    Parameters:
    -----------
    scores : pd.Series
        The scores used to determine long and short positions.
    mu_implied : pd.Series
        The implied mean returns.
    scalefactor : int, optional
        A scaling factor for the expected returns (default is 252).

    Returns:
    --------
    P : pd.DataFrame
        The pick matrix representing the views.
    q : pd.Series
        The expected returns for the views.
    """
    # Compute quintile thresholds
    lower_threshold, upper_threshold = np.percentile(scores, [20, 80])

    # Identify long and short positions
    s_short = scores[scores <= lower_threshold]
    s_long = scores[scores >= upper_threshold]

    # Create long-short weights
    w_ls = pd.Series(0.0, index=scores.index)
    if not s_short.empty:
        w_ls[s_short.index] = -1 / len(s_short)
    if not s_long.empty:
        w_ls[s_long.index] = 1 / len(s_long)

    # Create the pick matrix (P)
    P = w_ls.to_frame().T.reset_index(drop=True)

    # Compute view portfolio expected return (q) by a long-short
    # portfolio of the best versus worst implied returns
    mu_low, mu_high = np.percentile(mu_implied, [20, 80])
    mu_short = mu_implied[mu_implied <= mu_low]
    mu_long = mu_implied[mu_implied >= mu_high]
    q = pd.Series([mu_long.mean() - mu_short.mean()]) * scalefactor

    return P, q


def view_from_scores_absolute(
    scores: pd.Series,
    mu_implied: pd.Series,
    scalefactor: int = 1,
) -> (pd.DataFrame, pd.Series):
    """
    Generate views based on full ranking of scores.

    Parameters:
    -----------
    scores: pd.Series
        The scores used to determine the ranking.
    mu_implied: pd.Series
        The implied mean returns.
    scalefactor: int, optional
        A scaling factor for the expected returns (default is 252).

    Returns:
    --------
    P: pd.DataFrame
        The pick matrix representing the views.
    q: pd.Series
        The expected returns for the views.
    """

    # Drop NaN values from scores
    scores_clean = scores.dropna()

    # Create the pick matrix
    P = pd.DataFrame(
        np.zeros((len(scores_clean), len(scores))),
        index=scores_clean.index,
        columns=scores.index
    )    
    # Set values to 1 for the scores that are not NaN
    for idx in scores_clean.index:
        P.loc[idx, idx] = 1

    if len(scores_clean) == len(mu_implied):

        # Rank the scores in descending order
        scores_rank = scores_clean.rank(ascending=False).astype(int)

        # Align the implied returns with the rank of the scores
        sorted_mu = mu_implied.sort_values(ascending=False)
        q = pd.Series(
            sorted_mu.iloc[scores_rank-1].values,  # Align ranks with sorted returns
            index=mu_implied.index
        ) * scalefactor

    else:
        # Compute the average mu_implied for each quantile
        thresholds = np.quantile(mu_implied, np.linspace(0, 1, len(scores_clean)))
        mu = pd.Series(index=scores_clean.sort_values().index, dtype=float)
        for i in range(len(thresholds)):
            if i == 0:
                idx = mu_implied <= thresholds[i+1]
            elif i == len(thresholds) - 1:
                idx = mu_implied >= thresholds[i-1]
            else:
                idx = (mu_implied >= thresholds[i-1]) & (mu_implied <= thresholds[i+1])
            mu.iloc[i] = mu_implied[idx].mean()

        q = mu[scores_clean.index] * scalefactor

    return P, q


def generate_views_from_scores(
    scores: pd.Series,
    mu_implied: pd.Series,
    method: str = 'quintile',
    scalefactor: int = 1,
) -> (pd.DataFrame, pd.Series):
    """
    Generate views based on scores using the specified method.

    Parameters:
    -----------
    scores: pd.Series
        The scores used to generate views.
    mu_implied : pd.Series
        The implied mean returns.
    method: str, optional
        The method to generate views ('quintile' or 'absolute').
        Default is 'quintile'.
    scalefactor: int, optional
        A scaling factor for the expected returns (default is 252).

    Returns:
    --------
    P: pd.DataFrame
        The pick matrix representing the views.
    q: pd.Series
        The expected returns for the views.
    """
    if method == 'quintile':
        return view_from_scores_quintile(scores, mu_implied, scalefactor)
    elif method == 'absolute':
        return view_from_scores_absolute(scores, mu_implied, scalefactor)
    else:
        raise ValueError("Invalid method. Use 'quintile' or 'absolute'.")

