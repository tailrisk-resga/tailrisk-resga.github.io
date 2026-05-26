# Reproducibility Notes

The public reproduction workflow is designed to start from prediction CSV files.
This avoids requiring users to reproduce GPU training, proprietary data access,
or long rolling-window experiments before they can verify the reported
evaluation procedures.

For training-based reproduction, record:

- Python and package versions.
- CUDA and GPU details, if applicable.
- Random seeds.
- Data vintage and preprocessing date.
- Exact rolling-window boundaries.
- Model hyperparameters.

The code should be run from the repository root or installed with
`pip install -e .`.
