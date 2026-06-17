"""
Example: Dumitrescu-Hurlin Panel Granger Causality Test

Demonstrates the test using simulated panel data.
"""

import numpy as np
import pandas as pd
from dumitrescu_hurlin import dumitrescu_hurlin_test, print_results


def generate_panel_data(N=10, T=50, seed=42, causal=True):
    """
    Generate simulated panel data.

    If causal=True, x Granger-causes y for some entities.
    If causal=False, x and y are independent.
    """
    np.random.seed(seed)
    records = []

    for i in range(1, N + 1):
        x = np.zeros(T)
        y = np.zeros(T)

        for t in range(1, T):
            x[t] = 0.5 * x[t - 1] + np.random.normal(0, 1)

            if causal and i <= N // 2:
                # x Granger-causes y for the first half of entities
                y[t] = 0.3 * y[t - 1] + 0.5 * x[t - 1] + np.random.normal(0, 1)
            else:
                # No causality
                y[t] = 0.3 * y[t - 1] + np.random.normal(0, 1)

        for t in range(T):
            records.append({"entity": i, "time": t + 1, "y": y[t], "x": x[t]})

    return pd.DataFrame(records)


def main():
    # ---- Example 1: Causality present ----
    print("\n" + "#" * 60)
    print("# Example 1: x Granger-causes y (for some entities)")
    print("#" * 60 + "\n")

    data = generate_panel_data(N=10, T=50, causal=True)
    results = dumitrescu_hurlin_test(
        data, y_var="y", x_var="x", entity_var="entity", time_var="time", lags=1
    )
    print_results(results, y_var="y", x_var="x")

    # ---- Example 2: No causality ----
    print("\n" + "#" * 60)
    print("# Example 2: No Granger causality")
    print("#" * 60 + "\n")

    data_no_cause = generate_panel_data(N=10, T=50, causal=False)
    results_no = dumitrescu_hurlin_test(
        data_no_cause, y_var="y", x_var="x", entity_var="entity", time_var="time", lags=1
    )
    print_results(results_no, y_var="y", x_var="x")

    # ---- Example 3: Multiple lags ----
    print("\n" + "#" * 60)
    print("# Example 3: Testing with 2 lags")
    print("#" * 60 + "\n")

    results_2lag = dumitrescu_hurlin_test(
        data, y_var="y", x_var="x", entity_var="entity", time_var="time", lags=2
    )
    print_results(results_2lag, y_var="y", x_var="x")

    # ---- Example 4: Bidirectional test ----
    print("\n" + "#" * 60)
    print("# Example 4: Bidirectional causality test")
    print("#" * 60 + "\n")

    print("Direction: x -> y")
    results_xy = dumitrescu_hurlin_test(
        data, y_var="y", x_var="x", entity_var="entity", time_var="time", lags=1
    )
    print_results(results_xy, y_var="y", x_var="x")

    print("\nDirection: y -> x")
    results_yx = dumitrescu_hurlin_test(
        data, y_var="x", x_var="y", entity_var="entity", time_var="time", lags=1
    )
    print_results(results_yx, y_var="x", x_var="y")


if __name__ == "__main__":
    main()
