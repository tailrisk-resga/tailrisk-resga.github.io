import os
import time

import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset

from data.dataset import (
    CrossSectional_Dataset,
    DInformer_Dataset,
    EInformer_Dataset,
    Informer_Dataset,
    Pointwise_Dataset,
    ReSGA_Dataset,
    Temporal_Dataset,
)
from models import (
    DInformer,
    DLinear,
    EInformer,
    GRU,
    Informer,
    LANN,
    Linear,
    LSTM,
    NN,
    ReSGA,
    SGA,
)
from training.losses import L_FZ0


MODEL_REGISTRY = {
    "Linear": Linear,
    "NN": NN,
    "LANN": LANN,
    "DLinear": DLinear,
    "LSTM": LSTM,
    "GRU": GRU,
    "Informer": Informer,
    "EInformer": EInformer,
    "DInformer": DInformer,
    "SGA": SGA,
    "ReSGA": ReSGA,
}

DATASET_REGISTRY = {
    "Linear": Pointwise_Dataset,
    "NN": Pointwise_Dataset,
    "LANN": Temporal_Dataset,
    "DLinear": Temporal_Dataset,
    "LSTM": Temporal_Dataset,
    "GRU": Temporal_Dataset,
    "Informer": Informer_Dataset,
    "EInformer": EInformer_Dataset,
    "DInformer": DInformer_Dataset,
    "SGA": CrossSectional_Dataset,
    "ReSGA": ReSGA_Dataset,
}


def _loader_runtime_options(num_workers: int, device: torch.device) -> dict:
    return {
        "num_workers": num_workers,
        "persistent_workers": num_workers > 0,
        "pin_memory": device.type == "cuda",
    }


class Trainer:
    def __init__(self, args):
        self.args = args
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        if args.log_dir is not None:
            os.makedirs(args.log_dir, exist_ok=True)

        self.device = self._resolve_device()
        self.network = self.select_model(args.model).to(self.device)
        self.optim = Adam(
            self.network.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        self.best_score = 1e20
        self.early_stopping_count = 0

    def _resolve_device(self) -> torch.device:
        if self.args.use_gpu and torch.cuda.is_available():
            torch.cuda.manual_seed(self.args.seed)
            torch.cuda.set_device(self.args.cuda)
            return torch.device(f"cuda:{self.args.cuda}")
        return torch.device("cpu")

    def select_model(self, model):
        try:
            return MODEL_REGISTRY[model](self.args)
        except KeyError as exc:
            raise ValueError(f"No such model: {model}") from exc

    def select_dataset(self, model, data_dir, start_time, valid_time, test_time, end_time):
        try:
            dataset_cls = DATASET_REGISTRY[model]
        except KeyError as exc:
            raise ValueError(f"No such model: {model}") from exc
        return dataset_cls.split(data_dir, start_time, valid_time, test_time, end_time)

    def _make_loader(self, dataset, shuffle: bool, num_workers: int) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=shuffle,
            **_loader_runtime_options(num_workers, self.device),
        )

    def _prepare_inputs(self, inputs):
        inputs = [x.to(self.device) for x in inputs]
        if self.args.model == "LANN":
            inputs = [x.view(x.shape[0], -1) for x in inputs]
        return inputs

    def _predict_batch(self, inputs):
        inputs = self._prepare_inputs(inputs)
        return self.network(*inputs)

    def _fz_loss(self, y, v, e):
        y = y.to(self.device).view_as(v)
        return L_FZ0(self.args.alpha, y, v, e)

    def _checkpoint_path(self) -> str:
        return os.path.join(self.args.log_dir, "network_best.pth")

    def _load_best_checkpoint(self):
        try:
            checkpoint = torch.load(
                self._checkpoint_path(),
                map_location=self.device,
                weights_only=True,
            )
        except TypeError:
            checkpoint = torch.load(self._checkpoint_path(), map_location=self.device)
        self.network.load_state_dict(checkpoint)
        self.network.eval()

    def load_data_valid_only(self, data_dir, valid_time, test_time, num_workers=4):
        valid_time = pd.to_datetime(valid_time)
        test_time = pd.to_datetime(test_time)
        _, valid_dataset, _ = self.select_dataset(
            self.args.model,
            data_dir,
            valid_time,
            valid_time,
            test_time,
            test_time,
        )
        self.valid_loader = self._make_loader(valid_dataset, shuffle=False, num_workers=num_workers)

    def load_data_test_only(self, data_dir, test_time, end_time, num_workers=4):
        test_time = pd.to_datetime(test_time)
        end_time = pd.to_datetime(end_time)
        _, _, test_dataset = self.select_dataset(
            self.args.model,
            data_dir,
            test_time,
            test_time,
            test_time,
            end_time,
        )
        self.test_loader = self._make_loader(test_dataset, shuffle=False, num_workers=num_workers)

    def load_data(self, data_dir, start_time, valid_time, test_time, end_time, num_workers=4):
        start_time = pd.to_datetime(start_time)
        valid_time = pd.to_datetime(valid_time)
        test_time = pd.to_datetime(test_time)
        end_time = pd.to_datetime(end_time)

        train_dataset, valid_dataset, test_dataset = self.select_dataset(
            self.args.model,
            data_dir,
            start_time,
            valid_time,
            test_time,
            end_time,
        )
        train_dataset, valid_dataset = self._maybe_subsample_train_valid(
            train_dataset,
            valid_dataset,
        )

        self.train_loader = self._make_loader(train_dataset, shuffle=True, num_workers=num_workers)
        self.valid_loader = self._make_loader(valid_dataset, shuffle=True, num_workers=num_workers)
        self.test_loader = self._make_loader(test_dataset, shuffle=False, num_workers=num_workers)

    def _maybe_subsample_train_valid(self, train_dataset, valid_dataset):
        train_frac = self.args.train_frac
        if train_frac is None:
            train_frac = 1.0
        if not (0 < train_frac <= 1.0):
            raise ValueError(f"train_frac must be in (0, 1], got {train_frac}")
        if train_frac == 1.0:
            return train_dataset, valid_dataset

        n_train = len(train_dataset)
        n_valid = len(valid_dataset)
        m_train = max(1, int(round(n_train * train_frac)))
        m_valid = max(1, int(round(n_valid * train_frac)))

        rng = np.random.default_rng(self.args.seed)
        idx_train = np.sort(rng.choice(n_train, size=m_train, replace=False))
        idx_valid = np.sort(rng.choice(n_valid, size=m_valid, replace=False))

        print(f"[Train subsample] train_frac={train_frac:.2f}, {m_train}/{n_train} samples kept.")
        print(f"[Valid subsample] train_frac={train_frac:.2f}, {m_valid}/{n_valid} samples kept.")
        return Subset(train_dataset, idx_train), Subset(valid_dataset, idx_valid)

    def train(self, epoch: int = 1000):
        self.network.train()
        trainable_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        print(f"Trainable parameters: {trainable_params}")

        for i in range(epoch):
            start_time = time.time()
            train_losses = []

            for _, _, *inputs, y in self.train_loader:
                v, e = self._predict_batch(inputs)
                fz_loss = self._fz_loss(y, v, e)
                l1_loss = sum(torch.sum(torch.abs(param)) for param in self.network.parameters())
                loss = fz_loss + self.args.lambda_l1 * l1_loss

                train_losses.append(loss.item())
                self.update_params(
                    self.optim,
                    loss,
                    networks=[self.network],
                    retain_graph=False,
                    clip_value=self.args.clip_value,
                )

            train_loss = torch.tensor(train_losses).mean()
            elapsed = time.time() - start_time
            print(f"Epoch {i + 1}/{epoch} Training Loss {train_loss:.6f}  Time Consume {elapsed:.3f}")

            if (i + 1) % 3 == 0:
                self.validate(i)

            if self.early_stopping_count >= self.args.patience:
                del self.train_loader
                torch.cuda.empty_cache()
                break

        if not os.path.exists(self._checkpoint_path()):
            torch.save(self.network.state_dict(), self._checkpoint_path())

    def validate(self, i):
        self.network.eval()
        with torch.no_grad():
            valid_losses = []
            for _, _, *inputs, y in self.valid_loader:
                v, e = self._predict_batch(inputs)
                valid_losses.append(self._fz_loss(y, v, e).item())
            valid_loss = torch.tensor(valid_losses).mean()

        print("-" * 60)
        print(f"Validation {(i + 1) // 3} Loss {valid_loss.item():.6f}")
        if valid_loss.item() < self.best_score:
            print("update model")
            self.best_score = valid_loss.item()
            torch.save(self.network.state_dict(), self._checkpoint_path())
            self.early_stopping_count = 0
        else:
            self.early_stopping_count += 1
        print("-" * 60)
        self.network.train()

    def update_params(self, optim, loss, networks, retain_graph=False, clip_value=10):
        optim.zero_grad()
        loss.backward(retain_graph=retain_graph)
        if clip_value:
            for net in networks:
                torch.nn.utils.clip_grad_norm_(net.parameters(), clip_value)
        optim.step()

    def save_predictions(self, save_dir: str, split: str = "test"):
        os.makedirs(save_dir, exist_ok=True)
        loaders, out_name = self._prediction_loaders(split)
        self._load_best_checkpoint()

        iter_results = []
        with torch.no_grad():
            for loader in loaders:
                for stock_ids, dates, *inputs, y in loader:
                    v, e = self._predict_batch(inputs)
                    iter_results.append(self._prediction_frame(stock_ids, dates, y, v, e))

        inference_table = pd.concat(iter_results, axis=0).reset_index(drop=True)
        inference_table = inference_table.drop_duplicates(subset=["id", "eom"])
        out_path = os.path.join(save_dir, out_name)
        inference_table.to_csv(out_path, index=False)
        print(f"[Saved] {split} predictions -> {out_path}")

    def _prediction_loaders(self, split: str):
        if split == "valid":
            loaders = [getattr(self, "valid_loader", None)]
            out_name = "valid.csv"
            loader_names = ["valid_loader"]
        elif split == "test":
            loaders = [getattr(self, "test_loader", None)]
            out_name = "test.csv"
            loader_names = ["test_loader"]
        elif split == "insample":
            loaders = [getattr(self, "train_loader", None), getattr(self, "valid_loader", None)]
            out_name = "insample.csv"
            loader_names = ["train_loader", "valid_loader"]
        else:
            raise ValueError("split must be one of {'valid', 'test', 'insample'}")

        missing = [name for name, loader in zip(loader_names, loaders) if loader is None]
        if missing:
            raise ValueError(f"Missing loader(s): {missing}")
        return loaders, out_name

    @staticmethod
    def _batch_dates(dates, n_ids: int):
        dates = list(dates)
        if len(dates) == 1:
            return np.array([dates[0]] * n_ids)
        return np.array(dates).reshape(-1)

    def _prediction_frame(self, stock_ids, dates, y, v, e, mask: str | None = None):
        ids = np.array(stock_ids).reshape(-1)
        data = {
            "id": ids,
            "eom": self._batch_dates(dates, len(ids)),
            "v": v.detach().cpu().numpy().reshape(-1),
            "e": e.detach().cpu().numpy().reshape(-1),
            "y": y.detach().cpu().numpy().reshape(-1),
        }
        if mask is not None:
            data["mask"] = mask
        return pd.DataFrame(data)


__all__ = ["Trainer"]
