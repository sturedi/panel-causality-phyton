"""
Dumitrescu-Hurlin Panel Granger Causality Test

Reference:
    Dumitrescu, E.-I., & Hurlin, C. (2012). Testing for Granger non-causality
    in heterogeneous panels. Economic Modelling, 29(4), 1450-1460.
"""

import numpy as np
import pandas as pd
from scipy import stats


def dumitrescu_hurlin_test(data, y_var, x_var, entity_var, time_var, lags=1):
    """
    Perform the Dumitrescu-Hurlin (2012) panel Granger causality test.

    Tests the null hypothesis that x does not Granger-cause y for any
    cross-sectional unit in the panel, against the alternative that x
    Granger-causes y for at least some units.

    Parameters
    ----------
    data : pd.DataFrame
        Panel dataset in long format.
    y_var : str
        Name of the dependent variable column.
    x_var : str
        Name of the independent (potentially causal) variable column.
    entity_var : str
        Name of the cross-sectional entity identifier column.
    time_var : str
        Name of the time identifier column.
    lags : int, default 1
        Number of lags (K) to include in the test regression.

    Returns
    -------
    dict
        w_bar : float
            Average Wald statistic across all entities.
        z_bar : float
            Standardized Z-bar statistic (asymptotic, T -> inf first).
        p_value_z_bar : float
            Two-sided p-value for Z-bar.
        z_bar_tilde : float or None
            Semi-asymptotic Z-bar tilde statistic (finite T adjustment).
            None if T is too small relative to K.
        p_value_z_bar_tilde : float or None
            Two-sided p-value for Z-bar tilde.
            None if T is too small relative to K.
        individual_wald : dict
            Wald statistics for each individual entity.
        n_entities : int
            Number of cross-sectional entities.
        lags : int
            Number of lags used.
    """
    K = lags
    if K < 1:
        raise ValueError("Number of lags must be at least 1.")

    entities = data[entity_var].unique()
    N = len(entities)
    if N < 1:
        raise ValueError("No entities found in the data.")

    wald_stats = {}
    T_values = []

    for entity in entities:
        entity_data = data[data[entity_var] == entity].sort_values(time_var)
        y = entity_data[y_var].values.astype(float)
        x = entity_data[x_var].values.astype(float)
        T_total = len(y)

        if T_total <= 2 * K + 1:
            raise ValueError(
                f"Entity '{entity}' has {T_total} observations, but at least "
                f"{2 * K + 2} are required for {K} lag(s)."
            )

        T_i = T_total - K  # usable observations
        T_values.append(T_i)
        y_dep = y[K:]

        # Restricted model: y_t = alpha + sum(gamma_k * y_{t-k}) + eps
        X_r = np.ones((T_i, K + 1))
        for k in range(1, K + 1):
            X_r[:, k] = y[K - k : T_total - k]

        # Unrestricted model: y_t = alpha + sum(gamma_k * y_{t-k}) + sum(beta_k * x_{t-k}) + eps
        X_u = np.ones((T_i, 2 * K + 1))
        for k in range(1, K + 1):
            X_u[:, k] = y[K - k : T_total - k]
            X_u[:, K + k] = x[K - k : T_total - k]

        # OLS estimation
        beta_r = np.linalg.lstsq(X_r, y_dep, rcond=None)[0]
        RSS_r = np.sum((y_dep - X_r @ beta_r) ** 2)

        beta_u = np.linalg.lstsq(X_u, y_dep, rcond=None)[0]
        RSS_u = np.sum((y_dep - X_u @ beta_u) ** 2)

        # Wald statistic: W_i = K * F_i, where F_i ~ F(K, T_i - 2K - 1)
        df_u = T_i - 2 * K - 1
        F_i = ((RSS_r - RSS_u) / K) / (RSS_u / df_u)
        W_i = K * F_i

        wald_stats[entity] = W_i

    wald_array = np.array(list(wald_stats.values()))

    # ---- W-bar statistic ----
    W_bar = np.mean(wald_array)

    # ---- Z-bar statistic (asymptotic: T -> inf, then N -> inf) ----
    # Under H0 as T -> inf: W_i -> chi2(K), so E[W_i] = K, Var[W_i] = 2K
    Z_bar = np.sqrt(N / (2.0 * K)) * (W_bar - K)
    p_z_bar = 2.0 * (1.0 - stats.norm.cdf(abs(Z_bar)))

    # ---- Z-bar tilde statistic (semi-asymptotic: finite T) ----
    # Uses exact moments of W_i/K ~ F(K, T_i - 2K - 1)
    # Handles unbalanced panels by computing per-entity moments
    Z_bar_tilde = None
    p_z_bar_tilde = None

    E_Wi_list = []
    Var_Wi_list = []
    valid = True

    for T_i in T_values:
        d2 = T_i - 2 * K - 1  # denominator df of F distribution
        if d2 <= 4:
            valid = False
            break
        # E[W_i] = K * d2 / (d2 - 2)
        E_Wi = K * d2 / (d2 - 2.0)
        # Var[W_i] = K^2 * Var[F(K, d2)] = K^2 * 2*d2^2*(K+d2-2) / (K*(d2-2)^2*(d2-4))
        Var_Wi = (
            2.0 * K * d2 ** 2 * (K + d2 - 2.0)
            / ((d2 - 2.0) ** 2 * (d2 - 4.0))
        )
        E_Wi_list.append(E_Wi)
        Var_Wi_list.append(Var_Wi)

    if valid and len(E_Wi_list) == N:
        E_Wbar = np.mean(E_Wi_list)
        Var_Wbar = np.mean(Var_Wi_list) / N  # Var(W_bar) = (1/N^2) * sum(Var_Wi)
        # Correction: Var(W_bar) = (1/N^2) * sum(Var_Wi) = mean(Var_Wi) / N
        if Var_Wbar > 0:
            Z_bar_tilde = (W_bar - E_Wbar) / np.sqrt(Var_Wbar)
            p_z_bar_tilde = 2.0 * (1.0 - stats.norm.cdf(abs(Z_bar_tilde)))

    return {
        "w_bar": W_bar,
        "z_bar": Z_bar,
        "p_value_z_bar": p_z_bar,
        "z_bar_tilde": Z_bar_tilde,
        "p_value_z_bar_tilde": p_z_bar_tilde,
        "individual_wald": wald_stats,
        "n_entities": N,
        "lags": K,
    }


def print_results(results, y_var="Y", x_var="X"):
    """
    Print formatted results of the Dumitrescu-Hurlin test.

    Parameters
    ----------
    results : dict
        Output from dumitrescu_hurlin_test().
    y_var : str
        Name of the dependent variable (for display).
    x_var : str
        Name of the independent variable (for display).
    """
    print("=" * 60)
    print("Dumitrescu-Hurlin Panel Granger Causality Test")
    print("=" * 60)
    print(f"H0: {x_var} does not Granger-cause {y_var} (for all entities)")
    print(f"H1: {x_var} Granger-causes {y_var} (for at least some entities)")
    print("-" * 60)
    print(f"Number of entities (N): {results['n_entities']}")
    print(f"Number of lags (K):     {results['lags']}")
    print("-" * 60)
    print(f"W-bar statistic:        {results['w_bar']:.4f}")
    print(f"Z-bar statistic:        {results['z_bar']:.4f}")
    print(f"Z-bar p-value:          {results['p_value_z_bar']:.4f}")
    print("-" * 60)
    if results["z_bar_tilde"] is not None:
        print(f"Z-bar tilde statistic:  {results['z_bar_tilde']:.4f}")
        print(f"Z-bar tilde p-value:    {results['p_value_z_bar_tilde']:.4f}")
    else:
        print("Z-bar tilde: Not available (T too small relative to K)")
    print("=" * 60)

    print("\nIndividual Wald Statistics:")
    print("-" * 30)
    for entity, w in results["individual_wald"].items():
        print(f"  Entity {entity}: {w:.4f}")
    print()
