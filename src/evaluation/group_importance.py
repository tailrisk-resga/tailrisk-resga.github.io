import os

import numpy as np
import pandas as pd
import torch


def evaluate_group_importance(
    trainer,
    groups: dict,
    feature_names: list,
    save_dir: str,
    save_name: str = "group_importance.csv",
):
    """Evaluate predictions after masking each characteristic group."""

    trainer._load_best_checkpoint()

    fname2idx = {name: i for i, name in enumerate(feature_names)}
    group2idx = {
        group: [fname2idx[name] for name in names if name in fname2idx]
        for group, names in groups.items()
    }
    scenarios = [("ALL", None)] + list(group2idx.items())

    all_results = []
    with torch.no_grad():
        for scen_name, zero_idx in scenarios:
            scen_results = []
            for batch in trainer.test_loader:
                stock_ids, dates = batch[0], batch[1]
                *payload, y = batch[2:-1], batch[-1]
                inputs = _masked_inputs(
                    payload=payload,
                    n_features=len(feature_names),
                    zero_idx=zero_idx,
                    device=trainer.device,
                    flatten_lann=trainer.args.model == "LANN",
                )
                v, e = trainer.network(*inputs)
                scen_results.append(_prediction_frame(stock_ids, dates, y, v, e, mask=scen_name))

            scen_table = pd.concat(scen_results, axis=0).reset_index(drop=True)
            scen_table = scen_table.drop_duplicates(subset=["id", "eom", "mask"])
            all_results.append(scen_table)

    out_table = pd.concat(all_results, axis=0).reset_index(drop=True)
    out_table = out_table.drop_duplicates(subset=["id", "eom", "mask"])

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_name)
    out_table.to_csv(save_path, index=False)
    print(f"Group-importance predictions saved to: {save_path}")
    return out_table


def _masked_inputs(payload, n_features: int, zero_idx, device, flatten_lann: bool):
    flat_payload = []
    for item in payload:
        if isinstance(item, (list, tuple)):
            flat_payload.extend(list(item))
        else:
            flat_payload.append(item)

    x_pos = None
    x = None
    for i, tensor in enumerate(flat_payload):
        if torch.is_tensor(tensor) and tensor.shape[-1] == n_features:
            x_pos = i
            x = tensor.to(device)
            break
    if x is None:
        raise ValueError("Error: X is None")

    inputs = []
    for i, item in enumerate(flat_payload):
        if i == x_pos:
            inputs.append(None)
        else:
            inputs.append(item.to(device) if torch.is_tensor(item) else item)

    if zero_idx is not None and len(zero_idx) > 0:
        x = x.clone()
        x[..., zero_idx] = 0.0
    inputs[x_pos] = x

    if flatten_lann:
        inputs = [
            tensor.view(tensor.shape[0], -1)
            if torch.is_tensor(tensor) and tensor.dim() == 3 else tensor
            for tensor in inputs
        ]
    return inputs


def _prediction_frame(stock_ids, dates, y, v, e, mask: str):
    ids = np.array(stock_ids).reshape(-1)
    return pd.DataFrame({
        "id": ids,
        "eom": _batch_dates(dates, len(ids)),
        "v": v.detach().cpu().numpy().reshape(-1),
        "e": e.detach().cpu().numpy().reshape(-1),
        "y": y.detach().cpu().numpy().reshape(-1),
        "mask": mask,
    })


def _batch_dates(dates, n_ids: int):
    dates = list(dates)
    if len(dates) == 1:
        return np.array([dates[0]] * n_ids)
    return np.array(dates).reshape(-1)


__all__ = ["evaluate_group_importance"]
