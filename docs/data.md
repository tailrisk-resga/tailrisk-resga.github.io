# Data Layout

The training pipeline expects raw country CSV files, processed panel data, and
generated sample tensors. For public evaluation from prediction results, raw
data is not required.

Recommended local layout:

```text
data/
  raw/
  interim/
  processed/
  samples/
    <country>/
      pointwise/
      temporal/
      cross_sectional/
      resga/
outputs/
  checkpoints/
  predictions/
  metrics/
  tables/
  figures/
```

Large files under `data/` and `outputs/` are ignored by Git. The empty directory
structure is kept with `.gitkeep` files. Small example files can be placed under
`examples/` if needed.
