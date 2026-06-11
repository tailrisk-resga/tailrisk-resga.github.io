# Expected Shortfall Prediction

This repository is the official implementation of "[ReSGA: A Large Tail Risk Model for Learning Value-at-Risk and Expected Shortfall](https://arxiv.org/abs/2606.04576)".

This repository contains research code for expected shortfall prediction. The
public reproduction path starts from model prediction CSV files, so readers can
reproduce evaluation tables and backtests without re-training the neural models
or accessing the original raw data.

## Repository Layout

```text
configs/         YAML configuration files for data, models, training, evaluation.
scripts/         Numbered command-line entry points for the reproducible workflow.
src/
  data/          Feature definitions, cleaning, sample generation, dataset access.
  models/        Linear/NN/sequence/Informer/SGA/ReSGA model definitions.
  training/      Trainer and losses.
  evaluation/    Prediction aggregation and evaluation utilities.
  utils/         Config, paths, device, seed, and IO helpers.
data/            Local data directory; raw/interim/processed/samples files are not committed.
outputs/         Checkpoints, predictions, metrics, tables, and figures.
```

Exploratory notebooks and machine-specific legacy scripts are excluded from the anonymous review release.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Optional dependencies are separated because some baselines need extra system
packages:

```bash
pip install -e ".[baselines,logging]"
```

## Reproducing From Prediction Results

Place prediction CSV files under a result directory such as:

```text
outputs/predictions/
  USA/
    ReSGA/
      1_16_10/
        seed_1/
          2014-01-01_2015-01-01/valid.csv
          2014-01-01_2015-01-01/test.csv
        seed_2/
          2014-01-01_2015-01-01/valid.csv
          2014-01-01_2015-01-01/test.csv
    GARCH/GARCH.csv
    GAS/GAS.csv
```

Each prediction file should contain at least:

```text
id,eom,v,e,y
```

where `v` is the predicted VaR, `e` is the predicted expected shortfall, and `y`
is the realized return or loss variable used in the paper.

Run a smoke aggregation and loss calculation:

```bash
python3 scripts/04_evaluate.py \
  --prediction-root outputs/predictions \
  --country USA \
  --output-dir outputs/metrics
```

By default this uses `outputs/metrics/USA/best_hyperparameters.csv` from
`03_tune_params.py` and keeps `GAS`/`GARCH` baseline predictions. Pass
`--all-hyperparameters` to inspect every trained hyperparameter.

The default test-set outputs are:

```text
outputs/metrics/USA/test_selected_hyperparameters_predictions_with_loss.csv
outputs/metrics/USA/test_selected_hyperparameters_loss_summary.csv
```

More detailed result schemas are documented in [docs/result_schema.md](docs/result_schema.md).

## Training From Raw Data

The training pipeline is driven by one YAML config instead of many duplicated
shell scripts. The default template is
[configs/pipeline/raw_to_predictions.yaml](configs/pipeline/raw_to_predictions.yaml).

Prepare raw data and generated samples:

```bash
python3 scripts/01_prepare_data.py --config configs/pipeline/raw_to_predictions.yaml
```

Inspect the planned training runs without launching them:

```bash
python3 scripts/02_train.py --config configs/pipeline/raw_to_predictions.yaml --dry-run
```

Train the configured models:

```bash
python3 scripts/02_train.py --config configs/pipeline/raw_to_predictions.yaml
```

Select hyperparameters using validation predictions:

```bash
python3 scripts/03_tune_params.py \
  --prediction-root outputs/predictions \
  --country USA \
  --output-dir outputs/metrics
```

`GAS` and `GARCH` have no validation split. Their single CSV files are treated
as test predictions and are excluded from hyperparameter tuning.

Run a focused subset, for example one model/seed/GPU:

```bash
python3 scripts/02_train.py \
  --config configs/pipeline/raw_to_predictions.yaml \
  --model ReSGA \
  --seed 1 \
  --device cuda \
  --gpu-id 0
```

See [docs/raw_to_predictions.md](docs/raw_to_predictions.md) for the full
workflow and output layout.

## Model Organization

The proposed public boundary is:

- `src/models`: one file per neural model, including ReSGA.
- `src/models/baselines.py`: competing methods such as GARCH and GAS.
- `src/evaluation`: model-agnostic code that consumes prediction CSVs.

This keeps the paper's contribution separate from comparison methods while
making all models evaluable through the same output schema.

The supported neural model names are `Linear`, `NN`, `LANN`, `DLinear`, `LSTM`,
`GRU`, `Informer`, `EInformer`, `DInformer`, `SGA`, and `ReSGA`.

## Data

Raw data can be public if you provide it with the release. Large generated files
and result directories are ignored by Git by default. See [docs/data.md](docs/data.md)
for the expected directory layout and [docs/reproducibility.md](docs/reproducibility.md)
for reproducibility notes.

## Development Checks

```bash
PYTHONPYCACHEPREFIX=/private/tmp/es_pycache python3 -m compileall src scripts
pytest
```
