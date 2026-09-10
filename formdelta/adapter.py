from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from formdelta.reference import regression_metrics, slice_metrics


@dataclass
class AdapterConfig:
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.05
    lr: float = 3e-4
    weight_decay: float = 1e-5
    batch_size: int = 1024
    epochs: int = 200
    patience: int = 30
    huber_beta: float = 0.05
    seed: int = 20260528
    device: str = "cuda"
    train_fraction: float = 1.0
    train_subset_seed: int = 20260529


class ResidualAdapter(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dim = in_dim
        for _ in range(max(1, num_layers)):
            layers.extend(
                [
                    nn.Linear(dim, hidden_dim),
                    nn.SiLU(),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout),
                ]
            )
            dim = hidden_dim
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def load_adapter_frame(calibrated_predictions: Path, feature_cache: Path) -> pd.DataFrame:
    x_cols = [f"x_Z{z}" for z in range(1, 119)]
    pred = pd.read_csv(calibrated_predictions, keep_default_na=False, low_memory=False)
    if set(x_cols).issubset(pred.columns):
        frame = pred.copy()
        frame[x_cols] = frame[x_cols].apply(pd.to_numeric, errors="coerce")
        needs_backfill = bool(frame[x_cols].isna().any(axis=None))
    else:
        frame = pred.copy()
        needs_backfill = True
    if needs_backfill:
        feature_cols = ["id", "cif_path", *x_cols]
        feats = pd.read_csv(feature_cache, keep_default_na=False, usecols=feature_cols)
        frame = frame.merge(feats, on=["id", "cif_path"], how="left", suffixes=("", "_cache"))
    for col in x_cols:
        cache_col = f"{col}_cache"
        if col in frame.columns:
            base = pd.to_numeric(frame[col], errors="coerce")
            if cache_col in frame.columns:
                fallback = pd.to_numeric(frame[cache_col], errors="coerce")
                base = base.where(np.isfinite(base), fallback)
                frame = frame.drop(columns=[cache_col])
            frame[col] = base
        elif cache_col in frame.columns:
            frame[col] = pd.to_numeric(frame[cache_col], errors="coerce")
            frame = frame.drop(columns=[cache_col])
    if frame[x_cols].isna().any(axis=None):
        raise RuntimeError("Missing composition features for adapter training.")
    return frame


def build_features(frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    x_cols = [f"x_Z{z}" for z in range(1, 119)]
    features = frame[x_cols].to_numpy(dtype=np.float32)
    scalar_cols = [
        "model_energy_per_atom",
        "pred_element_reference",
        "pred_affine_energy_composition",
        "natoms",
        "nelements",
    ]
    scalars = []
    names: list[str] = []
    for col in scalar_cols:
        if col in frame.columns:
            value = frame[col].to_numpy(dtype=np.float32)
            if col == "natoms":
                value = np.log1p(value)
            scalars.append(value[:, None])
            names.append(col)
    if scalars:
        features = np.concatenate([features, *scalars], axis=1)
    names = [*x_cols, *names]
    return features, names


def standardize_train_val_test(x: np.ndarray, train_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x[train_mask].mean(axis=0, keepdims=True)
    std = x[train_mask].std(axis=0, keepdims=True)
    std[std < 1e-8] = 1.0
    return ((x - mean) / std).astype(np.float32), mean.squeeze(0), std.squeeze(0)


def run_adapter_training(
    *,
    calibrated_predictions: Path,
    feature_cache: Path,
    out_dir: Path,
    config: AdapterConfig,
    baseline_col: str = "pred_element_reference",
) -> dict[str, object]:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = load_adapter_frame(calibrated_predictions, feature_cache)
    frame["split"] = frame["split"].astype(str)
    if baseline_col not in frame.columns:
        raise KeyError(f"baseline column {baseline_col!r} not found in {calibrated_predictions}")

    full_train_mask = frame["split"].to_numpy() == "train"
    train_mask = full_train_mask.copy()
    val_mask = frame["split"].to_numpy() == "val"
    test_mask = frame["split"].to_numpy() == "test"
    if config.train_fraction <= 0.0 or config.train_fraction > 1.0:
        raise ValueError("train_fraction must be in (0, 1]")
    train_subset_summary: dict[str, object] = {
        "fraction": float(config.train_fraction),
        "seed": int(config.train_subset_seed),
        "full_train_rows": int(np.sum(full_train_mask)),
        "used_train_rows": int(np.sum(train_mask)),
        "enabled": False,
    }
    if config.train_fraction < 1.0:
        full_train_indices = np.flatnonzero(full_train_mask)
        n_keep = max(1, int(round(len(full_train_indices) * float(config.train_fraction))))
        rng_subset = np.random.default_rng(int(config.train_subset_seed))
        permutation = rng_subset.permutation(full_train_indices)
        keep = np.sort(permutation[:n_keep])
        train_mask = np.zeros(len(frame), dtype=bool)
        train_mask[keep] = True
        train_subset_summary.update(
            {
                "enabled": True,
                "used_train_rows": int(n_keep),
                "dropped_train_rows": int(len(full_train_indices) - n_keep),
            }
        )

    x_raw, feature_names = build_features(frame)
    x, x_mean, x_std = standardize_train_val_test(x_raw, train_mask)

    y_true = frame["y_true"].to_numpy(dtype=np.float32)
    baseline = frame[baseline_col].to_numpy(dtype=np.float32)
    residual = y_true - baseline
    residual_mean = float(residual[train_mask].mean())
    residual_std = float(residual[train_mask].std())
    if residual_std < 1e-8:
        residual_std = 1.0
    target = ((residual - residual_mean) / residual_std).astype(np.float32)

    device = torch.device(config.device if torch.cuda.is_available() and config.device.startswith("cuda") else "cpu")
    model = ResidualAdapter(x.shape[1], config.hidden_dim, config.num_layers, config.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, config.epochs), eta_min=config.lr * 0.05)
    loss_fn = nn.SmoothL1Loss(beta=config.huber_beta)

    x_tensor = torch.from_numpy(x)
    target_tensor = torch.from_numpy(target)
    train_indices = np.flatnonzero(train_mask)
    val_indices = np.flatnonzero(val_mask)

    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    best_epoch = -1
    history: list[dict[str, float | int]] = []
    stale = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        rng = np.random.default_rng(config.seed + epoch)
        rng.shuffle(train_indices)
        train_losses = []
        for start in range(0, len(train_indices), config.batch_size):
            batch_idx = train_indices[start : start + config.batch_size]
            xb = x_tensor[batch_idx].to(device)
            yb = target_tensor[batch_idx].to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_pred_norm = model(x_tensor[val_indices].to(device)).detach().cpu().numpy()
        val_pred = baseline[val_indices] + val_pred_norm * residual_std + residual_mean
        val_mae = float(np.mean(np.abs(val_pred - y_true[val_indices])) * 1000.0)
        history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_mae_mev_atom": val_mae})
        if val_mae < best_val:
            best_val = val_mae
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch % 10 == 0 or epoch == 1:
            print(f"[adapter] epoch={epoch} train_loss={np.mean(train_losses):.5f} val_mae={val_mae:.3f}", flush=True)
        if stale >= config.patience:
            print(f"[adapter] early stop at epoch {epoch}; best epoch {best_epoch}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    preds_norm = []
    with torch.no_grad():
        for start in range(0, len(frame), config.batch_size):
            preds_norm.append(model(x_tensor[start : start + config.batch_size].to(device)).detach().cpu().numpy())
    pred_norm = np.concatenate(preds_norm)
    frame["adapter_delta"] = pred_norm * residual_std + residual_mean
    frame["pred_fe_adapter"] = baseline + frame["adapter_delta"].to_numpy(dtype=np.float32)
    frame["adapter_error"] = frame["pred_fe_adapter"] - frame["y_true"]
    frame["adapter_abs_error_mev_atom"] = frame["adapter_error"].abs() * 1000.0

    metrics: dict[str, object] = {
        "config": {
            **config.__dict__,
            "device_used": str(device),
            "baseline_col": baseline_col,
            "calibrated_predictions": str(calibrated_predictions),
            "feature_cache": str(feature_cache),
            "best_epoch": best_epoch,
            "residual_mean": residual_mean,
            "residual_std": residual_std,
            "feature_names": feature_names,
            "train_subset": train_subset_summary,
        },
        "baseline": {},
        "adapter": {},
    }
    for split, group in frame.groupby("split"):
        metrics["baseline"][split] = regression_metrics(group["y_true"].to_numpy(), group[baseline_col].to_numpy())
        metrics["adapter"][split] = {
            "overall": regression_metrics(group["y_true"].to_numpy(), group["pred_fe_adapter"].to_numpy()),
            "slices": slice_metrics(group, "pred_fe_adapter", "y_true"),
        }

    frame.to_csv(out_dir / "fe_adapter_predictions.csv", index=False)
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    torch.save(
        {
            "model": model.state_dict(),
            "config": config.__dict__,
            "feature_names": feature_names,
            "x_mean": x_mean,
            "x_std": x_std,
            "residual_mean": residual_mean,
            "residual_std": residual_std,
            "baseline_col": baseline_col,
        },
        out_dir / "fe_adapter.pt",
    )
    return metrics
