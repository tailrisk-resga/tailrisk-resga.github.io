from __future__ import annotations

import json
import random
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from utils.device import describe_runtime


SEQUENCE_MODELS = {"LANN", "DLinear", "LSTM", "GRU", "Informer", "EInformer", "DInformer"}
PANEL_MODELS = {"SGA"}
RESGA_MODELS = {"ReSGA"}
PANEL_DATA_MODELS = {"Linear", "NN"}


def set_global_seed(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def model_hyperparameter_name(model: str, args: Namespace) -> str:
    if model == "Linear":
        return str(args.lambda_l1)
    if model in {"NN", "LANN", "DLinear"}:
        return str(args.hidden_dim)
    if model in {"LSTM", "GRU"}:
        return f"{args.num_layers}_{args.hidden_dim}"
    if model == "Informer":
        return (
            f"{args.d_model}_{args.n_heads}_{args.e_layers}_{args.d_layers}_"
            f"{args.d_ff}_{args.factor}_{args.distil}"
        )
    if model == "EInformer":
        return f"{args.d_model}_{args.n_heads}_{args.e_layers}_{args.d_ff}_{args.factor}_{args.distil}"
    if model == "DInformer":
        return f"{args.d_model}_{args.n_heads}_{args.d_layers}_{args.d_ff}_{args.factor}_{args.distil}"
    if model in {"SGA", "ReSGA"}:
        return f"{args.num_layers}_{args.hidden_dim}_{args.K}"
    raise ValueError(f"Unsupported model: {model}")


def data_source_for_model(model: str, country: str, data_dir: Path) -> str | Path:
    if model in PANEL_DATA_MODELS:
        return data_dir / "samples" / country / "pointwise"
    if model in SEQUENCE_MODELS:
        return data_dir / "samples" / country / "temporal"
    if model in PANEL_MODELS:
        return data_dir / "samples" / country / "cross_sectional"
    if model in RESGA_MODELS:
        return data_dir / "samples" / country / "resga"
    raise ValueError(f"Unsupported model: {model}")


def build_training_args(
    model: str,
    seed: int,
    country: str,
    training: dict[str, Any],
    model_config: dict[str, Any],
    device_config,
    log_dir: Path,
) -> Namespace:
    params = {
        "model": model,
        "exp_name": f"{country}_{model}_seed{seed}",
        "epochs": 10000,
        "batch_size": 25600,
        "patience": 5,
        "num_workers": 4,
        "seed": seed,
        "use_gpu": device_config.use_gpu,
        "cuda": device_config.cuda,
        "feature_dim": 153,
        "tol": 1e-4,
        "rolling": True,
        "rolling_steps": 1,
        "learning_rate": 1e-4,
        "weight_decay": 0.0,
        "lambda_l1": 0.0,
        "train_frac": 1.0,
        "alpha": 0.05,
        "clip_value": 10.0,
        "hidden_dim": 32,
        "num_layers": 3,
        "sequence_length": 12,
        "dropout": 0.1,
        "K": 10,
        "beta": 0.5,
        "d_model": 256,
        "n_heads": 8,
        "e_layers": 2,
        "d_layers": 1,
        "d_ff": 1024,
        "factor": 5,
        "embed": "fixed",
        "freq": "b",
        "activation": "gelu",
        "output_attention": False,
        "distil": False,
        "c_out": 2,
        "start_time": "1926-01-01",
        "valid_time": "1996-01-01",
        "test_time": "2014-01-01",
        "end_time": "2015-01-01",
        "log_dir": str(log_dir),
    }
    params.update(training or {})
    params.update(model_config or {})
    params.update(
        {
            "model": model,
            "seed": seed,
            "use_gpu": device_config.use_gpu,
            "cuda": device_config.cuda,
            "log_dir": str(log_dir),
        }
    )
    return Namespace(**params)


def save_run_metadata(log_dir: Path, config: dict[str, Any], args: Namespace) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "runtime": describe_runtime(),
        "args": vars(args),
        "config": config,
    }
    with (log_dir / "run_metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=str)
