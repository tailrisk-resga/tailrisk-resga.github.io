# Raw Data to Prediction Results

The training side of the project is now configuration-driven. The intended
flow is:

```text
raw CSV
  -> processed panel CSV
  -> pointwise / temporal / cross-sectional / ReSGA samples
  -> rolling-window model training
  -> checkpoints + valid/test prediction CSVs
```

## Main Config

Use [configs/pipeline/raw_to_predictions.yaml](../configs/pipeline/raw_to_predictions.yaml)
as the starting point. It defines:

- `project`: data and output roots.
- `countries`: countries to run.
- `seeds`: random seeds.
- `runtime`: device policy (`auto`, `cpu`, or `cuda`) and GPU id.
- `data`: preprocessing and sample-building parameters.
- `training`: shared training defaults.
- `models`: model list and per-model parameter overrides.

The same parameters are also split into smaller files for readability:

- `configs/data/us.yaml`
- `configs/model/all_models.yaml`
- `configs/model/resga.yaml`
- `configs/train/default.yaml`
- `configs/eval/default.yaml`

The pipeline config is the executable, merged version used by the scripts.

## Commands

Prepare data:

```bash
python3 scripts/01_prepare_data.py --config configs/pipeline/raw_to_predictions.yaml
```

Train all configured models/seeds/rolling windows:

```bash
python3 scripts/02_train.py --config configs/pipeline/raw_to_predictions.yaml
```

Select each model's best hyperparameter using validation predictions:

```bash
python3 scripts/03_tune_params.py \
  --prediction-root outputs/predictions \
  --country USA \
  --output-dir outputs/metrics
```

`03_tune_params.py` only reads files named `valid.csv`. Single-output
baselines such as `GAS/GAS.csv` and `GARCH/GARCH.csv` do not have validation
results; they are treated as test predictions and are not tuned here.

Dry-run the training plan:

```bash
python3 scripts/02_train.py --config configs/pipeline/raw_to_predictions.yaml --dry-run
```

Run a subset:

```bash
python3 scripts/02_train.py \
  --config configs/pipeline/raw_to_predictions.yaml \
  --model ReSGA \
  --seed 1 \
  --device cuda \
  --gpu-id 0
```

Run both preparation and training:

```bash
python3 scripts/run_pipeline.py --config configs/pipeline/raw_to_predictions.yaml
```

## Output Layout

Data preparation writes samples to:

```text
data/samples/<country>/pointwise/<stock_id>_<date>.npy
  feature: float32 vector with shape (153,)
  label: float32 vector with shape (1,)
data/samples/<country>/temporal/<stock_id>_<date>.npy
  feature: float32 array with shape (sequence_length, 153)
  label: float32 vector with shape (1,)
data/samples/<country>/cross_sectional/<date>.npy
data/samples/<country>/resga/<YYYYMMDD>.pt
```

`Linear` and `NN` use the pointwise samples. `LANN`, `DLinear`, `LSTM`, `GRU`,
`Informer`, `EInformer`, and `DInformer` use temporal samples. `SGA` uses
cross-sectional samples, and `ReSGA` uses ReSGA samples.

Training outputs are written to:

```text
outputs/checkpoints/<country>/<model>/<hyperparameter>/seed_<seed>/<test_start>_<test_end>/
  network_best.pth
  run_metadata.json
outputs/predictions/<country>/<model>/<hyperparameter>/seed_<seed>/<test_start>_<test_end>/
  valid.csv
  test.csv
outputs/runs/<country>/<model>/<hyperparameter>/seed_<seed>/<test_start>_<test_end>/
  run_metadata.json
```

`run_metadata.json` stores the merged config, resolved arguments, torch/CUDA
runtime information, and timestamp for reproducibility.

Hyperparameter tuning writes:

```text
outputs/metrics/<country>/validation_predictions_with_loss.csv
outputs/metrics/<country>/validation_loss_summary.csv
outputs/metrics/<country>/best_hyperparameters.csv
```

Final evaluation uses `best_hyperparameters.csv` by default, so each neural
model is evaluated only with its validation-selected hyperparameter. Baselines
such as `GAS` and `GARCH` are kept because they do not have hyperparameters or
validation results.

Final evaluation writes:

```text
outputs/metrics/<country>/<split>_selected_hyperparameters_predictions_with_loss.csv
outputs/metrics/<country>/<split>_selected_hyperparameters_loss_summary.csv
```

Use `--all-hyperparameters` to write the corresponding
`<split>_all_hyperparameters_*` diagnostic files.

## GPU Policy

Set `runtime.device` to:

- `auto`: use CUDA if available, otherwise CPU.
- `cuda`: require CUDA and fail if unavailable.
- `cpu`: force CPU.

Use `runtime.gpu_id` or `--gpu-id` to select the CUDA device. For cluster or
multi-screen workflows, prefer using `CUDA_VISIBLE_DEVICES` plus `--gpu-id 0`
rather than hard-coding different CUDA ids in separate scripts.
