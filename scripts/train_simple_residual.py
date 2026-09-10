#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from formdelta.adapter import AdapterConfig, run_adapter_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a composition residual FE adapter from calibrated predictions.")
    parser.add_argument("--calibrated-predictions", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--baseline-col", default="pred_element_reference")
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--huber-beta", type=float, default=0.05)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--train-subset-seed", type=int, default=20260529)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AdapterConfig(
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        dropout=float(args.dropout),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        patience=int(args.patience),
        huber_beta=float(args.huber_beta),
        seed=int(args.seed),
        device=str(args.device),
        train_fraction=float(args.train_fraction),
        train_subset_seed=int(args.train_subset_seed),
    )
    metrics = run_adapter_training(
        calibrated_predictions=args.calibrated_predictions.resolve(),
        feature_cache=args.feature_cache.resolve(),
        out_dir=args.out_dir.resolve(),
        config=config,
        baseline_col=str(args.baseline_col),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    test_mae = float(metrics["adapter"]["test"]["overall"]["mae_mev_atom"])
    val_mae = float(metrics["adapter"]["val"]["overall"]["mae_mev_atom"])
    print(
        f"adapter training complete | seed={args.seed} | val_mae={val_mae:.3f} meV/atom | "
        f"test_mae={test_mae:.3f} meV/atom | out_dir={args.out_dir.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
