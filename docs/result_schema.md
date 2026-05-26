# Prediction Result Schema

The public reproduction pipeline starts from prediction CSV files.

## Required Columns

| Column | Type | Description |
| --- | --- | --- |
| `id` | integer/string | Asset identifier. |
| `eom` | date-like string | End-of-month timestamp. |
| `v` | float | Predicted VaR / quantile. |
| `e` | float | Predicted expected shortfall. |
| `y` | float | Realized target return/loss. |

## Optional Columns

| Column | Type | Description |
| --- | --- | --- |
| `model` | string | Model name. Filled from folder names when absent. |
| `hyperparameter` | string | Hyperparameter folder name, filled from folder names when absent. |
| `seed` | integer/string | Random seed for repeated neural model runs. |
| `window` | string | Rolling test window folder, filled from folder names when absent. |
| `split` | string | Prediction split, usually `valid` or `test`. |
| `run` | string | Combined run id, usually `<hyperparameter>/seed_<seed>/<window>`. |
| `loss` | float | FZ loss. Recomputed by default for consistency. |

## Recommended Directory Layout

```text
outputs/predictions/<country>/<model>/<hyperparameter>/seed_<seed>/<test_start>_<test_end>/*.csv
```

Baselines with a single output file can use:

```text
outputs/predictions/<country>/<baseline>/<baseline>.csv
```

Single-file baselines such as `GAS` and `GARCH` do not have validation
predictions. They are treated as `test` predictions and are excluded from
hyperparameter tuning.
