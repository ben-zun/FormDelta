#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from formdelta.fegx import FEGX1Config, run_fegx1_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FE-GX-1 model-internal structure/risk/aux variants.")
    parser.add_argument("--calibrated-predictions", required=True)
    parser.add_argument("--feature-cache", required=True)
    parser.add_argument("--structure-cache", required=True)
    parser.add_argument("--coefficient-csv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--exp-name", default="fegx1a")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--pair-dim", type=int, default=16)
    parser.add_argument("--spacegroup-dim", type=int, default=16)
    parser.add_argument("--crystal-dim", type=int, default=4)
    parser.add_argument("--structure-hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--reference-lr", type=float, default=0.0)
    parser.add_argument(
        "--anchor-mode",
        choices=("fixed_anchor", "learnable_anchor", "stacking", "target_only"),
        default="fixed_anchor",
    )
    parser.add_argument("--learnable-anchor-init", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--huber-beta", type=float, default=0.05)
    parser.add_argument("--risk-weight", type=float, default=0.0)
    parser.add_argument("--risk-loss", choices=("none", "abs", "heteroscedastic"), default="none")
    parser.add_argument("--pos-weight", type=float, default=0.0)
    parser.add_argument("--near-weight", type=float, default=0.0)
    parser.add_argument("--near-zero-threshold", type=float, default=0.10)
    parser.add_argument("--rank-weight", type=float, default=0.0)
    parser.add_argument("--rank-margin-mev", type=float, default=20.0)
    parser.add_argument("--rank-pair-mode", choices=("all_pairs", "top1"), default="all_pairs")
    parser.add_argument("--rank-temperature", type=float, default=0.03)
    parser.add_argument("--rank-batch-size", type=int, default=1024)
    parser.add_argument("--rank-pairs-per-epoch", type=int, default=20000)
    parser.add_argument("--rank-max-pairs-per-formula", type=int, default=20000)
    parser.add_argument("--tail-weight-mode", choices=("none", "positive_binary", "ref", "ref_positive_binary"), default="none")
    parser.add_argument("--tail-max-weight", type=float, default=1.0)
    parser.add_argument("--tail-ref-quantile", type=float, default=0.90)
    parser.add_argument("--tail-positive-increment", type=float, default=0.5)
    parser.add_argument("--tail-binary-increment", type=float, default=0.3)
    parser.add_argument("--tail-ref-increment", type=float, default=0.5)
    parser.add_argument("--tail-l2-weight", type=float, default=0.0)
    parser.add_argument("--residual-head-type", choices=("mlp", "moe"), default="mlp")
    parser.add_argument("--num-experts", type=int, default=3)
    parser.add_argument("--router-hidden-dim", type=int, default=64)
    parser.add_argument("--router-temperature", type=float, default=1.0)
    parser.add_argument("--moe-load-balance-weight", type=float, default=0.0)
    parser.add_argument("--moe-entropy-weight", type=float, default=0.0)
    parser.add_argument("--moe-router-init", choices=("uniform", "expert0"), default="uniform")
    parser.add_argument("--moe-noise-std", type=float, default=0.0)
    parser.add_argument("--router-anchor-weight", type=float, default=0.0)
    parser.add_argument("--router-anchor-risk-quantile", type=float, default=0.95)
    parser.add_argument("--router-anchor-target-expert", type=int, default=2)
    parser.add_argument("--use-risk-residual-branch", action="store_true")
    parser.add_argument("--risk-branch-hidden-dim", type=int, default=64)
    parser.add_argument("--risk-branch-gate-bias", type=float, default=-2.0)
    parser.add_argument("--risk-branch-anchor-weight", type=float, default=0.0)
    parser.add_argument("--tail-feature-cache", default="")
    parser.add_argument("--use-tail-features", action="store_true")
    parser.add_argument("--tail-feature-cols", default="")
    parser.add_argument("--tail-feature-dim", type=int, default=16)
    parser.add_argument("--tail-feature-hidden-dim", type=int, default=32)
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--init-transfer-mode", choices=("all", "body_only"), default="all")
    parser.add_argument("--element-env-cache", default="")
    parser.add_argument("--use-element-env", action="store_true")
    parser.add_argument("--element-env-dim", type=int, default=16)
    parser.add_argument("--element-env-hidden-dim", type=int, default=64)
    parser.add_argument("--element-env-delta-weight", type=float, default=1.0)
    parser.add_argument("--element-env-to-residual-head", action="store_true")
    parser.add_argument("--pair-env-cache", default="")
    parser.add_argument("--use-pair-env", action="store_true")
    parser.add_argument("--pair-env-dim", type=int, default=16)
    parser.add_argument("--pair-env-hidden-dim", type=int, default=64)
    parser.add_argument("--pair-env-delta-weight", type=float, default=1.0)
    parser.add_argument("--max-pair-env-pairs", type=int, default=64)
    parser.add_argument(
        "--pair-env-ablation",
        choices=("none", "no_geometry", "no_identity", "no_same_pair", "no_bond_count"),
        default="none",
    )
    parser.add_argument("--shell-pair-env-cache", default="")
    parser.add_argument("--use-shell-pair-env", action="store_true")
    parser.add_argument("--shell-pair-env-dim", type=int, default=16)
    parser.add_argument("--shell-pair-env-hidden-dim", type=int, default=64)
    parser.add_argument("--shell-pair-env-delta-weight", type=float, default=1.0)
    parser.add_argument("--max-shell-pair-env-pairs", type=int, default=64)
    parser.add_argument("--chem-pair-env-cache", default="")
    parser.add_argument("--use-chem-pair-env", action="store_true")
    parser.add_argument("--chem-pair-env-dim", type=int, default=16)
    parser.add_argument("--chem-pair-env-hidden-dim", type=int, default=64)
    parser.add_argument("--chem-pair-env-delta-weight", type=float, default=1.0)
    parser.add_argument("--max-chem-pair-env-pairs", type=int, default=64)
    parser.add_argument(
        "--chem-pair-env-mode",
        choices=("covnorm", "chemprop", "full", "noraw"),
        default="full",
    )
    parser.add_argument("--use-dual-foundation", action="store_true")
    parser.add_argument("--dual-foundation-calibrated-predictions", default="")
    parser.add_argument("--dual-foundation-dim", type=int, default=16)
    parser.add_argument("--dual-foundation-hidden-dim", type=int, default=64)
    parser.add_argument(
        "--dual-foundation-ablation",
        choices=(
            "none",
            "no_aux_energy",
            "no_aux_reference",
            "no_disagreement",
            "energy_only",
            "no_energy_signal",
            "aux_energy_only",
            "reference_only",
        ),
        default="none",
    )
    parser.add_argument("--use-dual-reference-gate", action="store_true")
    parser.add_argument("--dual-reference-gate-hidden-dim", type=int, default=16)
    parser.add_argument("--dual-reference-gate-init-weight", type=float, default=0.90)
    parser.add_argument("--dual-reference-gate-prior-weight", type=float, default=0.0)
    parser.add_argument("--dual-reference-gate-entropy-weight", type=float, default=0.0)
    parser.add_argument("--dual-reference-gate-use-raw-dual-stats", action="store_true")
    parser.add_argument("--use-primary-missing-branch", action="store_true")
    parser.add_argument("--primary-missing-branch-hidden-dim", type=int, default=128)
    parser.add_argument("--primary-train-mask-prob", type=float, default=0.0)
    parser.add_argument("--use-grouped-structure-adapter", action="store_true")
    parser.add_argument("--grouped-structure-group-dim", type=int, default=16)
    parser.add_argument("--grouped-structure-group-hidden-dim", type=int, default=32)
    parser.add_argument("--grouped-structure-context-hidden-dim", type=int, default=64)
    parser.add_argument("--grouped-structure-context-dim", type=int, default=32)
    parser.add_argument("--grouped-structure-interaction-dim", type=int, default=32)
    parser.add_argument("--grouped-structure-dropout", type=float, default=0.05)
    parser.add_argument("--grouped-structure-magnitude-bias", type=float, default=-1.5)
    parser.add_argument("--grouped-structure-freeze-base", action="store_true")
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--train-subset-seed", type=int, default=20260529)
    parser.add_argument("--train-sampling-mode", choices=("shuffle", "hybrid_coverage"), default="shuffle")
    parser.add_argument("--hybrid-sampling-primary-ratio", type=float, default=0.70)
    parser.add_argument("--hybrid-sampling-group-col", default="chemistry_group")
    parser.add_argument("--validation-bucket-col", default="validation_bucket")
    parser.add_argument("--seed", type=int, default=20260528)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = FEGX1Config(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pair_dim=args.pair_dim,
        spacegroup_dim=args.spacegroup_dim,
        crystal_dim=args.crystal_dim,
        structure_hidden_dim=args.structure_hidden_dim,
        lr=args.lr,
        reference_lr=args.reference_lr,
        anchor_mode=args.anchor_mode,
        learnable_anchor_init=args.learnable_anchor_init,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        huber_beta=args.huber_beta,
        risk_weight=args.risk_weight,
        risk_loss=args.risk_loss,
        pos_weight=args.pos_weight,
        near_weight=args.near_weight,
        near_zero_threshold=args.near_zero_threshold,
        rank_weight=args.rank_weight,
        rank_margin_mev=args.rank_margin_mev,
        rank_pair_mode=args.rank_pair_mode,
        rank_temperature=args.rank_temperature,
        rank_batch_size=args.rank_batch_size,
        rank_pairs_per_epoch=args.rank_pairs_per_epoch,
        rank_max_pairs_per_formula=args.rank_max_pairs_per_formula,
        tail_weight_mode=args.tail_weight_mode,
        tail_max_weight=args.tail_max_weight,
        tail_ref_quantile=args.tail_ref_quantile,
        tail_positive_increment=args.tail_positive_increment,
        tail_binary_increment=args.tail_binary_increment,
        tail_ref_increment=args.tail_ref_increment,
        tail_l2_weight=args.tail_l2_weight,
        residual_head_type=args.residual_head_type,
        num_experts=args.num_experts,
        router_hidden_dim=args.router_hidden_dim,
        router_temperature=args.router_temperature,
        moe_load_balance_weight=args.moe_load_balance_weight,
        moe_entropy_weight=args.moe_entropy_weight,
        moe_router_init=args.moe_router_init,
        moe_noise_std=args.moe_noise_std,
        router_anchor_weight=args.router_anchor_weight,
        router_anchor_risk_quantile=args.router_anchor_risk_quantile,
        router_anchor_target_expert=args.router_anchor_target_expert,
        use_risk_residual_branch=args.use_risk_residual_branch,
        risk_branch_hidden_dim=args.risk_branch_hidden_dim,
        risk_branch_gate_bias=args.risk_branch_gate_bias,
        risk_branch_anchor_weight=args.risk_branch_anchor_weight,
        tail_feature_cache=args.tail_feature_cache,
        use_tail_features=args.use_tail_features,
        tail_feature_cols=args.tail_feature_cols,
        tail_feature_dim=args.tail_feature_dim,
        tail_feature_hidden_dim=args.tail_feature_hidden_dim,
        init_checkpoint=args.init_checkpoint,
        init_transfer_mode=args.init_transfer_mode,
        element_env_cache=args.element_env_cache,
        use_element_env=args.use_element_env,
        element_env_dim=args.element_env_dim,
        element_env_hidden_dim=args.element_env_hidden_dim,
        element_env_delta_weight=args.element_env_delta_weight,
        element_env_to_residual_head=args.element_env_to_residual_head,
        pair_env_cache=args.pair_env_cache,
        use_pair_env=args.use_pair_env,
        pair_env_dim=args.pair_env_dim,
        pair_env_hidden_dim=args.pair_env_hidden_dim,
        pair_env_delta_weight=args.pair_env_delta_weight,
        max_pair_env_pairs=args.max_pair_env_pairs,
        pair_env_ablation=args.pair_env_ablation,
        shell_pair_env_cache=args.shell_pair_env_cache,
        use_shell_pair_env=args.use_shell_pair_env,
        shell_pair_env_dim=args.shell_pair_env_dim,
        shell_pair_env_hidden_dim=args.shell_pair_env_hidden_dim,
        shell_pair_env_delta_weight=args.shell_pair_env_delta_weight,
        max_shell_pair_env_pairs=args.max_shell_pair_env_pairs,
        chem_pair_env_cache=args.chem_pair_env_cache,
        use_chem_pair_env=args.use_chem_pair_env,
        chem_pair_env_dim=args.chem_pair_env_dim,
        chem_pair_env_hidden_dim=args.chem_pair_env_hidden_dim,
        chem_pair_env_delta_weight=args.chem_pair_env_delta_weight,
        max_chem_pair_env_pairs=args.max_chem_pair_env_pairs,
        chem_pair_env_mode=args.chem_pair_env_mode,
        use_dual_foundation=args.use_dual_foundation,
        dual_foundation_calibrated_predictions=args.dual_foundation_calibrated_predictions,
        dual_foundation_dim=args.dual_foundation_dim,
        dual_foundation_hidden_dim=args.dual_foundation_hidden_dim,
        dual_foundation_ablation=args.dual_foundation_ablation,
        use_dual_reference_gate=args.use_dual_reference_gate,
        dual_reference_gate_hidden_dim=args.dual_reference_gate_hidden_dim,
        dual_reference_gate_init_weight=args.dual_reference_gate_init_weight,
        dual_reference_gate_prior_weight=args.dual_reference_gate_prior_weight,
        dual_reference_gate_entropy_weight=args.dual_reference_gate_entropy_weight,
        dual_reference_gate_use_raw_dual_stats=args.dual_reference_gate_use_raw_dual_stats,
        use_primary_missing_branch=args.use_primary_missing_branch,
        primary_missing_branch_hidden_dim=args.primary_missing_branch_hidden_dim,
        primary_train_mask_prob=args.primary_train_mask_prob,
        use_grouped_structure_adapter=args.use_grouped_structure_adapter,
        grouped_structure_group_dim=args.grouped_structure_group_dim,
        grouped_structure_group_hidden_dim=args.grouped_structure_group_hidden_dim,
        grouped_structure_context_hidden_dim=args.grouped_structure_context_hidden_dim,
        grouped_structure_context_dim=args.grouped_structure_context_dim,
        grouped_structure_interaction_dim=args.grouped_structure_interaction_dim,
        grouped_structure_dropout=args.grouped_structure_dropout,
        grouped_structure_magnitude_bias=args.grouped_structure_magnitude_bias,
        grouped_structure_freeze_base=args.grouped_structure_freeze_base,
        train_fraction=args.train_fraction,
        train_subset_seed=args.train_subset_seed,
        train_sampling_mode=args.train_sampling_mode,
        hybrid_sampling_primary_ratio=args.hybrid_sampling_primary_ratio,
        hybrid_sampling_group_col=args.hybrid_sampling_group_col,
        validation_bucket_col=args.validation_bucket_col,
        seed=args.seed,
        device=args.device,
        exp_name=args.exp_name,
    )
    metrics = run_fegx1_training(
        calibrated_predictions=Path(args.calibrated_predictions),
        feature_cache=Path(args.feature_cache),
        structure_cache=Path(args.structure_cache),
        coefficient_csv=Path(args.coefficient_csv),
        out_dir=Path(args.out_dir),
        config=config,
    )
    out_path = Path(args.out_dir) / "metrics.json"
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
    test = metrics[args.exp_name].get("test", {}).get("overall", {})
    base = metrics["baseline"].get("test", {})
    print(
        f"{args.exp_name} complete | "
        f"baseline test MAE={base.get('mae_mev_atom', float('nan')):.3f} meV/atom | "
        f"test MAE={test.get('mae_mev_atom', float('nan')):.3f} meV/atom"
    )


if __name__ == "__main__":
    main()
