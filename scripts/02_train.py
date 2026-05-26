from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pandas as pd
from dateutil.relativedelta import relativedelta

from utils.config import deep_update, load_config
from utils.paths import ProjectPaths
from utils.device import resolve_device
from utils.experiment import (
    build_training_args,
    data_source_for_model,
    model_hyperparameter_name,
    save_run_metadata,
    set_global_seed,
)
from training.trainer import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ES prediction models from YAML config.")
    parser.add_argument("--config", required=True, help="Path to a YAML training config.")
    parser.add_argument("--model", action="append", help="Run only selected model. Can be repeated.")
    parser.add_argument("--country", action="append", help="Run only selected country. Can be repeated.")
    parser.add_argument("--seed", action="append", type=int, help="Run only selected seed. Can be repeated.")
    parser.add_argument("--device", default=None, help="Override config device: auto, cpu, or cuda.")
    parser.add_argument("--gpu-id", type=int, default=None, help="Override config gpu_id.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs without training.")
    return parser.parse_args()


def _rolling_windows(training: dict) -> list[tuple[str, str, str, str]]:
    steps = int(training.get("rolling_steps", 1))
    start_valid = pd.to_datetime(training.get("valid_time", "1996-01-01"))
    start_test = pd.to_datetime(training.get("test_time", "2014-01-01"))
    start_end = pd.to_datetime(training.get("end_time", "2015-01-01"))
    start_time = training.get("start_time", "1926-01-01")
    return [
        (
            start_time,
            (start_valid + relativedelta(years=i)).strftime("%Y-%m-%d"),
            (start_test + relativedelta(years=i)).strftime("%Y-%m-%d"),
            (start_end + relativedelta(years=i)).strftime("%Y-%m-%d"),
        )
        for i in range(steps)
    ]


def main() -> None:
    cli = parse_args()
    config = load_config(cli.config)
    project_cfg = config.get("project", {})
    paths = ProjectPaths.from_args(
        root=project_cfg.get("root"),
        data_dir=project_cfg.get("data_dir", "data"),
        prediction_root=project_cfg.get("prediction_root", "outputs/predictions"),
        output_dir=project_cfg.get("output_dir", "outputs"),
    )

    training = deepcopy(config.get("training", {}))
    runtime_cfg = config.get("runtime", {})
    device_name = cli.device or runtime_cfg.get("device", "auto")
    gpu_id = cli.gpu_id if cli.gpu_id is not None else int(runtime_cfg.get("gpu_id", 0))
    device_config = resolve_device(device_name, gpu_id=gpu_id)

    countries = cli.country or config.get("countries", ["USA"])
    seeds = cli.seed or config.get("seeds", [42])
    model_entries = config.get("models", [])
    if not model_entries:
        raise ValueError("Config must define a non-empty models list.")

    for country in countries:
        for model_entry in model_entries:
            model = model_entry["name"] if isinstance(model_entry, dict) else str(model_entry)
            if cli.model and model not in cli.model:
                continue
            model_overrides = model_entry.get("params", {}) if isinstance(model_entry, dict) else {}
            model_training = deep_update(training, model_overrides)

            for seed in seeds:
                for start_time, valid_time, test_time, end_time in _rolling_windows(model_training):
                    base_args = build_training_args(
                        model=model,
                        seed=seed,
                        country=country,
                        training=model_training,
                        model_config={},
                        device_config=device_config,
                        log_dir=Path("."),
                    )
                    hyperparameter = model_hyperparameter_name(model, base_args)
                    run_dir = paths.output_dir / "runs" / country / model / hyperparameter / f"seed_{seed}" / f"{test_time}_{end_time}"
                    checkpoint_dir = paths.output_dir / "checkpoints" / country / model / hyperparameter / f"seed_{seed}" / f"{test_time}_{end_time}"
                    prediction_dir = paths.output_dir / "predictions" / country / model / hyperparameter / f"seed_{seed}" / f"{test_time}_{end_time}"
                    args = build_training_args(
                        model=model,
                        seed=seed,
                        country=country,
                        training={
                            **model_training,
                            "start_time": start_time,
                            "valid_time": valid_time,
                            "test_time": test_time,
                            "end_time": end_time,
                        },
                        model_config={},
                        device_config=device_config,
                        log_dir=checkpoint_dir,
                    )

                    data_source = data_source_for_model(model, country, paths.data_dir)

                    print(
                        f"[train] country={country} model={model} seed={seed} "
                        f"window={test_time}_{end_time} device={device_config.device}"
                    )
                    print(f"        run_dir={run_dir}")
                    print(f"        checkpoint_dir={checkpoint_dir}")
                    print(f"        prediction_dir={prediction_dir}")
                    print(f"        data={data_source if isinstance(data_source, Path) else 'processed dataframe'}")
                    if cli.dry_run:
                        continue

                    set_global_seed(seed, deterministic=bool(runtime_cfg.get("deterministic", False)))
                    save_run_metadata(run_dir, config, args)
                    save_run_metadata(checkpoint_dir, config, args)
                    model_agent = Trainer(args)
                    model_agent.load_data(
                        data_dir=data_source,
                        start_time=start_time,
                        valid_time=valid_time,
                        test_time=test_time,
                        end_time=end_time,
                        num_workers=args.num_workers,
                    )
                    model_agent.train(epoch=args.epochs)
                    model_agent.save_predictions(save_dir=str(prediction_dir), split="valid")
                    model_agent.save_predictions(save_dir=str(prediction_dir))
                    del model_agent

    print("Training complete.")


if __name__ == "__main__":
    main()
