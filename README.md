# Dumitrescu-Hurlin Panel Granger Causality Test

Python implementation of the Dumitrescu & Hurlin (2012) panel Granger causality test.

## Reference

Dumitrescu, E.-I., & Hurlin, C. (2012). Testing for Granger non-causality in heterogeneous panels. *Economic Modelling*, 29(4), 1450-1460.

## Overview

The Dumitrescu-Hurlin test extends the standard Granger causality test to panel data settings. It allows for heterogeneous causal relationships across cross-sectional units.

- **Null Hypothesis (H0):** X does not Granger-cause Y for **any** entity in the panel.
- **Alternative Hypothesis (H1):** X Granger-causes Y for **at least some** entities.

The test computes three key statistics:

| Statistic | Description |
|-----------|-------------|
| **W-bar** | Average of individual Wald statistics across all entities |
| **Z-bar** | Standardized statistic (asymptotic, T → ∞ first, then N → ∞) |
| **Z-bar tilde** | Semi-asymptotic statistic with finite-T correction |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### In Python Code

```python
import pandas as pd
from dumitrescu_hurlin import dumitrescu_hurlin_test, print_results

# Load your panel data (long format)
data = pd.read_csv("your_data.csv")

# Run the test
results = dumitrescu_hurlin_test(
    data,
    y_var="gdp",          # Dependent variable
    x_var="investment",    # Independent variable
    entity_var="country",  # Cross-sectional identifier
    time_var="year",       # Time identifier
    lags=2                 # Number of lags
)

# Display results
print_results(results, y_var="gdp", x_var="investment")

# Access specific values
print(results["z_bar_tilde"])         # Z-bar tilde statistic
print(results["p_value_z_bar_tilde"]) # p-value
```

### From Command Line (with CSV)

```bash
# One direction
python example_csv.py --file data.csv --y gdp --x investment --entity country --time year --lags 2

# Bidirectional test
python example_csv.py --file data.csv --y gdp --x investment --entity country --time year --lags 2 --bidirectional
```

### Run Examples with Simulated Data

```bash
python example.py
```

## Data Format

The input data must be a pandas DataFrame in **long (stacked) panel format**:

| entity | time | y    | x    |
|--------|------|------|------|
| 1      | 2000 | 3.45 | 2.10 |
| 1      | 2001 | 3.67 | 2.34 |
| 1      | 2002 | 3.89 | 2.56 |
| 2      | 2000 | 4.12 | 1.98 |
| 2      | 2001 | 4.45 | 2.15 |
| 2      | 2002 | 4.78 | 2.30 |

## Output

The function returns a dictionary with:

| Key | Type | Description |
|-----|------|-------------|
| `w_bar` | float | Average Wald statistic |
| `z_bar` | float | Z-bar statistic |
| `p_value_z_bar` | float | Two-sided p-value for Z-bar |
| `z_bar_tilde` | float/None | Z-bar tilde statistic (finite-T correction) |
| `p_value_z_bar_tilde` | float/None | Two-sided p-value for Z-bar tilde |
| `individual_wald` | dict | Per-entity Wald statistics |
| `n_entities` | int | Number of cross-sectional entities |
| `lags` | int | Number of lags used |

## Features

- Supports **balanced and unbalanced** panels
- Computes both **asymptotic (Z-bar)** and **semi-asymptotic (Z-bar tilde)** statistics
- Reports **individual Wald statistics** for each entity
- Command-line interface for CSV data
- Bidirectional causality testing

## Requirements

- Python >= 3.8
- NumPy >= 1.20
- pandas >= 1.3
- SciPy >= 1.7

## License

MIT
