"""
Example: Using the Dumitrescu-Hurlin test with your own CSV data.

Expected CSV format (long/stacked panel format):
    entity,time,y,x
    1,2000,3.45,2.10
    1,2001,3.67,2.34
    ...
    2,2000,4.12,1.98
    2,2001,4.45,2.15
    ...

Usage:
    python example_csv.py --file data.csv --y gdp --x investment --entity country --time year --lags 2
"""

import argparse
import pandas as pd
from dumitrescu_hurlin import dumitrescu_hurlin_test, print_results


def main():
    parser = argparse.ArgumentParser(
        description="Dumitrescu-Hurlin Panel Granger Causality Test"
    )
    parser.add_argument("--file", required=True, help="Path to CSV file")
    parser.add_argument("--y", required=True, help="Dependent variable column name")
    parser.add_argument("--x", required=True, help="Independent variable column name")
    parser.add_argument("--entity", required=True, help="Entity identifier column name")
    parser.add_argument("--time", required=True, help="Time identifier column name")
    parser.add_argument("--lags", type=int, default=1, help="Number of lags (default: 1)")
    parser.add_argument(
        "--bidirectional", action="store_true",
        help="Also test reverse causality direction"
    )
    args = parser.parse_args()

    data = pd.read_csv(args.file)
    print(f"\nLoaded {len(data)} rows, {data[args.entity].nunique()} entities.\n")

    # Forward direction: x -> y
    print(f"Testing: {args.x} -> {args.y}")
    results = dumitrescu_hurlin_test(
        data,
        y_var=args.y,
        x_var=args.x,
        entity_var=args.entity,
        time_var=args.time,
        lags=args.lags,
    )
    print_results(results, y_var=args.y, x_var=args.x)

    # Reverse direction: y -> x
    if args.bidirectional:
        print(f"\nTesting reverse: {args.y} -> {args.x}")
        results_rev = dumitrescu_hurlin_test(
            data,
            y_var=args.x,
            x_var=args.y,
            entity_var=args.entity,
            time_var=args.time,
            lags=args.lags,
        )
        print_results(results_rev, y_var=args.x, x_var=args.y)


if __name__ == "__main__":
    main()
