from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from formdelta.adapter import ResidualAdapter, load_adapter_frame
from formdelta.reference import regression_metrics, slice_metrics


MAX_Z = 118
X_COLS = [f"x_Z{z}" for z in range(1, MAX_Z + 1)]


@dataclass
class FEGX0Config:
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.05
    lr: float = 3e-4
    reference_lr: float = 0.0
    weight_decay: float = 1e-5
    batch_size: int = 2048
    epochs: int = 200
    patience: int = 30
    huber_beta: float = 0.05
    seed: int = 20260528
    device: str = "cuda"


@dataclass
class FEGX1Config(FEGX0Config):
    pair_dim: int = 16
    spacegroup_dim: int = 16
    crystal_dim: int = 4
    structure_hidden_dim: int = 64
    risk_weight: float = 0.0
    risk_loss: str = "none"
    pos_weight: float = 0.0
    near_weight: float = 0.0
    near_zero_threshold: float = 0.10
    rank_weight: float = 0.0
    rank_margin_mev: float = 20.0
    rank_pair_mode: str = "all_pairs"
    rank_temperature: float = 0.03
    rank_batch_size: int = 1024
    rank_pairs_per_epoch: int = 20000
    rank_max_pairs_per_formula: int = 20000
    tail_weight_mode: str = "none"
    tail_max_weight: float = 1.0
    tail_ref_quantile: float = 0.90
    tail_positive_increment: float = 0.5
    tail_binary_increment: float = 0.3
    tail_ref_increment: float = 0.5
    tail_l2_weight: float = 0.0
    residual_head_type: str = "mlp"
    num_experts: int = 3
    router_hidden_dim: int = 64
    router_temperature: float = 1.0
    moe_load_balance_weight: float = 0.0
    moe_entropy_weight: float = 0.0
    moe_router_init: str = "uniform"
    moe_noise_std: float = 0.0
    router_anchor_weight: float = 0.0
    router_anchor_risk_quantile: float = 0.95
    router_anchor_target_expert: int = 2
    use_risk_residual_branch: bool = False
    risk_branch_hidden_dim: int = 64
    risk_branch_gate_bias: float = -2.0
    risk_branch_anchor_weight: float = 0.0
    tail_feature_cache: str = ""
    use_tail_features: bool = False
    tail_feature_cols: str = ""
    tail_feature_dim: int = 16
    tail_feature_hidden_dim: int = 32
    init_checkpoint: str = ""
    init_transfer_mode: str = "all"
    element_env_cache: str = ""
    use_element_env: bool = False
    element_env_dim: int = 16
    element_env_hidden_dim: int = 64
    element_env_delta_weight: float = 1.0
    element_env_to_residual_head: bool = False
    pair_env_cache: str = ""
    use_pair_env: bool = False
    pair_env_dim: int = 16
    pair_env_hidden_dim: int = 64
    pair_env_delta_weight: float = 1.0
    max_pair_env_pairs: int = 64
    pair_env_ablation: str = "none"
    shell_pair_env_cache: str = ""
    use_shell_pair_env: bool = False
    shell_pair_env_dim: int = 16
    shell_pair_env_hidden_dim: int = 64
    shell_pair_env_delta_weight: float = 1.0
    max_shell_pair_env_pairs: int = 64
    chem_pair_env_cache: str = ""
    use_chem_pair_env: bool = False
    chem_pair_env_dim: int = 16
    chem_pair_env_hidden_dim: int = 64
    chem_pair_env_delta_weight: float = 1.0
    max_chem_pair_env_pairs: int = 64
    chem_pair_env_mode: str = "full"
    use_dual_foundation: bool = False
    dual_foundation_calibrated_predictions: str = ""
    dual_foundation_dim: int = 16
    dual_foundation_hidden_dim: int = 64
    dual_foundation_ablation: str = "none"
    use_dual_reference_gate: bool = False
    dual_reference_gate_hidden_dim: int = 16
    dual_reference_gate_init_weight: float = 0.90
    dual_reference_gate_prior_weight: float = 0.0
    dual_reference_gate_entropy_weight: float = 0.0
    dual_reference_gate_use_raw_dual_stats: bool = False
    use_primary_missing_branch: bool = False
    primary_missing_branch_hidden_dim: int = 128
    primary_train_mask_prob: float = 0.0
    use_grouped_structure_adapter: bool = False
    grouped_structure_group_dim: int = 16
    grouped_structure_group_hidden_dim: int = 32
    grouped_structure_context_hidden_dim: int = 64
    grouped_structure_context_dim: int = 32
    grouped_structure_interaction_dim: int = 32
    grouped_structure_dropout: float = 0.05
    grouped_structure_magnitude_bias: float = -1.5
    grouped_structure_freeze_base: bool = False
    train_fraction: float = 1.0
    train_subset_seed: int = 20260529
    train_sampling_mode: str = "shuffle"
    hybrid_sampling_primary_ratio: float = 0.70
    hybrid_sampling_group_col: str = "chemistry_group"
    validation_bucket_col: str = "validation_bucket"
    anchor_mode: str = "fixed_anchor"
    learnable_anchor_init: float = 1.0
    exp_name: str = "fegx1"


class ElementReferenceLayer(nn.Module):
    def __init__(self, mu: np.ndarray, bias: float, *, trainable: bool) -> None:
        super().__init__()
        self.mu = nn.Parameter(torch.as_tensor(mu, dtype=torch.float32), requires_grad=trainable)
        self.bias = nn.Parameter(torch.tensor(float(bias), dtype=torch.float32), requires_grad=trainable)

    def forward(self, composition: torch.Tensor, foundation_energy: torch.Tensor) -> torch.Tensor:
        return foundation_energy - composition @ self.mu - self.bias


class AffineReferenceLayer(nn.Module):
    def __init__(self, alpha: float, beta: np.ndarray, bias: float, *, trainable: bool) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(float(alpha), dtype=torch.float32), requires_grad=trainable)
        self.beta = nn.Parameter(torch.as_tensor(beta, dtype=torch.float32), requires_grad=trainable)
        self.bias = nn.Parameter(torch.tensor(float(bias), dtype=torch.float32), requires_grad=trainable)

    def forward(self, composition: torch.Tensor, foundation_energy: torch.Tensor) -> torch.Tensor:
        return self.alpha * foundation_energy + composition @ self.beta + self.bias


class ResidualMoEHead(nn.Module):
    """Small soft-routed residual expert head for FE-GX-1F."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        *,
        num_experts: int,
        router_hidden_dim: int,
        router_temperature: float,
        router_init: str,
    ) -> None:
        super().__init__()
        if num_experts < 2:
            raise ValueError("ResidualMoEHead requires at least two experts")
        self.num_experts = int(num_experts)
        self.router_temperature = float(max(router_temperature, 1e-6))
        self.experts = nn.ModuleList(
            [ResidualAdapter(in_dim, hidden_dim, num_layers, dropout) for _ in range(self.num_experts)]
        )
        self.router = nn.Sequential(
            nn.Linear(in_dim, router_hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(router_hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(router_hidden_dim, self.num_experts),
        )
        self.reset_router(router_init)

    def reset_router(self, router_init: str) -> None:
        last = self.router[-1]
        if not isinstance(last, nn.Linear):
            return
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        if str(router_init).lower() == "expert0":
            with torch.no_grad():
                last.bias[0] = 2.0

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.router(x)
        weights = torch.softmax(logits / self.router_temperature, dim=1)
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
        pred = torch.sum(weights * expert_outputs, dim=1)
        return pred, weights, expert_outputs


ELEMENT_ENV_STATS = [
    "site_frac",
    "site_count_frac",
    "coord_mean",
    "coord_std",
    "coord_min",
    "coord_max",
    "bond_mean",
    "bond_std",
    "bond_min",
    "bond_max",
    "hetero_neighbor_frac",
    "neighbor_element_entropy",
]

PAIR_ENV_STATS = [
    "pair_frac",
    "bond_count_frac",
    "bond_mean",
    "bond_std",
    "bond_min",
    "bond_max",
    "same_pair",
]

SHELL_PAIR_ENV_STATS = [
    "pair_frac",
    "same_pair",
    "shell1_bond_count_frac",
    "shell1_bond_mean",
    "shell1_bond_std",
    "shell1_bond_min",
    "shell1_bond_max",
    "shell2_bond_count_frac",
    "shell2_bond_mean",
    "shell2_bond_std",
    "shell2_bond_min",
    "shell2_bond_max",
    "shell3_bond_count_frac",
    "shell3_bond_mean",
    "shell3_bond_std",
    "shell3_bond_min",
    "shell3_bond_max",
]

CHEM_PAIR_ENV_STATS = [
    "pair_frac",
    "bond_count_frac",
    "bond_mean",
    "bond_std",
    "bond_min",
    "bond_max",
    "same_pair",
    "bond_mean_cov_norm",
    "bond_std_cov_norm",
    "bond_min_cov_norm",
    "bond_max_cov_norm",
    "bond_mean_atomic_norm",
    "bond_std_atomic_norm",
    "bond_min_atomic_norm",
    "bond_max_atomic_norm",
    "electronegativity_diff",
    "atomic_radius_ratio",
    "covalent_radius_sum",
    "atomic_radius_sum",
    "period_diff",
    "group_diff",
    "same_period",
    "same_group",
    "metal_pair_type",
]

DUAL_FOUNDATION_STATS = [
    "aux_model_energy_per_atom",
    "aux_element_reference",
    "aux_affine_reference",
    "energy_diff",
    "energy_abs_diff",
    "element_reference_diff",
    "element_reference_abs_diff",
    "affine_reference_diff",
    "affine_reference_abs_diff",
    "aux_missing",
    "primary_missing",
]

DUAL_FOUNDATION_PASSTHROUGH_STATS = {"aux_missing", "primary_missing"}
DUAL_FOUNDATION_PRIMARY_DEPENDENT_STATS = {
    "energy_diff",
    "energy_abs_diff",
    "element_reference_diff",
    "element_reference_abs_diff",
    "affine_reference_diff",
    "affine_reference_abs_diff",
}

DUAL_FOUNDATION_ACTIVE_STATS_BY_MODE = {
    "none": set(DUAL_FOUNDATION_STATS),
    "no_aux_energy": {
        "aux_element_reference",
        "aux_affine_reference",
        "energy_diff",
        "energy_abs_diff",
        "element_reference_diff",
        "element_reference_abs_diff",
        "affine_reference_diff",
        "affine_reference_abs_diff",
        "aux_missing",
    },
    "no_aux_reference": {
        "aux_model_energy_per_atom",
        "energy_diff",
        "energy_abs_diff",
        "aux_missing",
    },
    "no_disagreement": {
        "aux_model_energy_per_atom",
        "aux_element_reference",
        "aux_affine_reference",
        "aux_missing",
    },
    "energy_only": {
        "aux_model_energy_per_atom",
        "energy_diff",
        "energy_abs_diff",
        "aux_missing",
    },
    "no_energy_signal": {
        "aux_element_reference",
        "aux_affine_reference",
        "element_reference_diff",
        "element_reference_abs_diff",
        "affine_reference_diff",
        "affine_reference_abs_diff",
        "aux_missing",
    },
    "aux_energy_only": {
        "aux_model_energy_per_atom",
        "aux_missing",
    },
    "reference_only": {
        "aux_element_reference",
        "aux_affine_reference",
        "element_reference_diff",
        "element_reference_abs_diff",
        "affine_reference_diff",
        "affine_reference_abs_diff",
        "aux_missing",
    },
}

GROUPED_STRUCTURE_GROUP_NAMES = ("scale", "shape", "symmetry", "coordination", "bond")


def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    if logits.numel() == 0:
        return logits
    transpose_dim = dim if dim >= 0 else logits.dim() + dim
    transposed = logits.transpose(transpose_dim, -1)
    original_shape = transposed.shape
    flattened = transposed.reshape(-1, original_shape[-1])
    sorted_logits, _ = torch.sort(flattened, dim=1, descending=True)
    cssv = torch.cumsum(sorted_logits, dim=1) - 1.0
    k = (
        torch.arange(1, sorted_logits.shape[1] + 1, device=logits.device, dtype=logits.dtype)
        .unsqueeze(0)
        .expand_as(sorted_logits)
    )
    support = (k * sorted_logits) > cssv
    support_size = support.sum(dim=1, keepdim=True).clamp(min=1)
    tau = cssv.gather(1, support_size - 1) / support_size.to(dtype=logits.dtype)
    output = torch.clamp(flattened - tau, min=0.0)
    return output.reshape(original_shape).transpose(transpose_dim, -1)


def make_grouped_structure_encoder(input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
    hidden = max(int(hidden_dim), int(output_dim))
    return nn.Sequential(
        nn.Linear(int(input_dim), hidden),
        nn.SiLU(),
        nn.LayerNorm(hidden),
        nn.Dropout(dropout),
        nn.Linear(hidden, int(output_dim)),
        nn.SiLU(),
        nn.LayerNorm(int(output_dim)),
    )


class GroupedStructureAdapter(nn.Module):
    def __init__(
        self,
        *,
        pair_dim: int,
        spacegroup_dim: int,
        crystal_dim: int,
        dual_stats_dim: int,
        group_dim: int,
        group_hidden_dim: int,
        context_hidden_dim: int,
        context_dim: int,
        interaction_dim: int,
        dropout: float,
        magnitude_bias: float,
    ) -> None:
        super().__init__()
        self.group_names = list(GROUPED_STRUCTURE_GROUP_NAMES)
        self.context_input_dim = 5 + int(pair_dim) + int(dual_stats_dim)
        self.context_encoder = nn.Sequential(
            nn.Linear(self.context_input_dim, int(context_hidden_dim)),
            nn.SiLU(),
            nn.LayerNorm(int(context_hidden_dim)),
            nn.Dropout(dropout),
            nn.Linear(int(context_hidden_dim), int(context_dim)),
            nn.SiLU(),
            nn.LayerNorm(int(context_dim)),
        )
        self.encoders = nn.ModuleDict(
            {
                "scale": make_grouped_structure_encoder(2, group_hidden_dim, group_dim, dropout),
                "shape": make_grouped_structure_encoder(6, group_hidden_dim, group_dim, dropout),
                "symmetry": make_grouped_structure_encoder(2 + spacegroup_dim + crystal_dim, group_hidden_dim, group_dim, dropout),
                "coordination": make_grouped_structure_encoder(5, group_hidden_dim, group_dim, dropout),
                "bond": make_grouped_structure_encoder(4, group_hidden_dim, group_dim, dropout),
            }
        )
        scorer_hidden = max(16, int(group_dim))
        self.group_scorers = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(int(group_dim) + int(context_dim), scorer_hidden),
                    nn.Tanh(),
                    nn.Linear(scorer_hidden, 1),
                )
                for name in self.group_names
            }
        )
        self.struct_proj = nn.Linear(int(group_dim), int(interaction_dim))
        self.context_proj = nn.Linear(int(context_dim), int(interaction_dim))
        delta_hidden = max(16, int(interaction_dim))
        self.delta_head = nn.Sequential(
            nn.Linear(int(group_dim) + int(interaction_dim), delta_hidden),
            nn.SiLU(),
            nn.LayerNorm(delta_hidden),
            nn.Dropout(dropout),
            nn.Linear(delta_hidden, 1),
        )
        self.magnitude_gate = nn.Sequential(
            nn.Linear(int(group_dim) + int(context_dim), max(16, int(context_dim) // 2)),
            nn.SiLU(),
            nn.Linear(max(16, int(context_dim) // 2), 1),
        )
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)
        nn.init.constant_(self.magnitude_gate[-1].bias, float(magnitude_bias))

    def forward(self, groups: dict[str, torch.Tensor], context: torch.Tensor) -> dict[str, torch.Tensor]:
        context_token = self.context_encoder(context)
        encoded_groups: list[torch.Tensor] = []
        group_scores: list[torch.Tensor] = []
        for name in self.group_names:
            token = self.encoders[name](groups[name])
            encoded_groups.append(token)
            group_scores.append(self.group_scorers[name](torch.cat([token, context_token], dim=1)))
        encoded = torch.stack(encoded_groups, dim=1)
        scores = torch.cat(group_scores, dim=1)
        weights = sparsemax(scores, dim=1)
        z_struct = torch.sum(weights.unsqueeze(-1) * encoded, dim=1)
        cross = self.struct_proj(z_struct) * self.context_proj(context_token)
        raw_delta = self.delta_head(torch.cat([z_struct, cross], dim=1)).squeeze(-1)
        magnitude = torch.sigmoid(self.magnitude_gate(torch.cat([z_struct, context_token], dim=1)).squeeze(-1))
        return {
            "delta": magnitude * raw_delta,
            "raw_delta": raw_delta,
            "magnitude": magnitude,
            "weights": weights,
            "z_struct": z_struct,
        }
for _active_stats in DUAL_FOUNDATION_ACTIVE_STATS_BY_MODE.values():
    _active_stats.add("primary_missing")


class ElementEnvEncoder(nn.Module):
    """Element-conditioned local-environment residual branch."""

    def __init__(self, *, stats_dim: int, env_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.element_embedding = nn.Embedding(MAX_Z + 1, env_dim, padding_idx=0)
        self.site_frac_index = 0
        in_dim = env_dim + stats_dim
        self.site_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, env_dim),
            nn.SiLU(),
            nn.LayerNorm(env_dim),
        )
        self.delta_head = nn.Sequential(
            nn.Linear(env_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)

    def forward(self, env_stats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, num_elements, _ = env_stats.shape
        ids = torch.arange(1, num_elements + 1, device=env_stats.device, dtype=torch.long)
        emb = self.element_embedding(ids).unsqueeze(0).expand(batch, -1, -1)
        site_repr = self.site_encoder(torch.cat([emb, env_stats], dim=2))
        weights = env_stats[:, :, self.site_frac_index].clamp_min(0.0)
        pooled = torch.sum(site_repr * weights.unsqueeze(-1), dim=1) / torch.clamp(weights.sum(dim=1, keepdim=True), min=1e-6)
        delta_norm = self.delta_head(pooled).squeeze(-1)
        return delta_norm, pooled


class PairEnvEncoder(nn.Module):
    """Low-rank element-pair local-environment residual branch."""

    def __init__(self, *, stats_dim: int, pair_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.pair_embedding = nn.Embedding(MAX_Z * MAX_Z, pair_dim)
        self.weight_index = 0
        in_dim = pair_dim + stats_dim
        self.pair_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pair_dim),
            nn.SiLU(),
            nn.LayerNorm(pair_dim),
        )
        self.delta_head = nn.Sequential(
            nn.Linear(pair_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)

    def forward(self, pair_ids: torch.Tensor, pair_stats: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if pair_ids.numel() == 0 or pair_stats.numel() == 0:
            batch = pair_stats.shape[0] if pair_stats.ndim else 0
            empty = torch.empty((batch, 0), device=pair_stats.device)
            return torch.zeros((batch,), device=pair_stats.device), empty
        pair_ids = pair_ids.clamp(0, MAX_Z * MAX_Z - 1)
        emb = self.pair_embedding(pair_ids)
        pair_repr = self.pair_encoder(torch.cat([emb, pair_stats], dim=2))
        weights = pair_stats[:, :, self.weight_index].clamp_min(0.0)
        pooled = torch.sum(pair_repr * weights.unsqueeze(-1), dim=1) / torch.clamp(weights.sum(dim=1, keepdim=True), min=1e-6)
        delta_norm = self.delta_head(pooled).squeeze(-1)
        return delta_norm, pooled


class ShellPairEnvEncoder(PairEnvEncoder):
    """Shell-aware element-pair environment residual branch."""


class ChemPairEnvEncoder(PairEnvEncoder):
    """Chemistry-normalized element-pair environment residual branch."""


class DualFoundationEncoder(nn.Module):
    """Internal token encoder for a second foundation energy signal."""

    def __init__(self, *, stats_dim: int, token_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(stats_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, token_dim),
            nn.SiLU(),
            nn.LayerNorm(token_dim),
        )

    def forward(self, stats: torch.Tensor) -> torch.Tensor:
        return self.encoder(stats)


class DualReferenceGate(nn.Module):
    """Conservative gate between primary and auxiliary formation references."""

    def __init__(
        self,
        *,
        in_dim: int,
        hidden_dim: int,
        init_weight: float,
        dropout: float,
    ) -> None:
        super().__init__()
        if not 0.0 < init_weight < 1.0:
            raise ValueError("dual_reference_gate_init_weight must be between 0 and 1")
        hidden_dim = max(4, int(hidden_dim))
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        init_logit = float(np.log(init_weight / (1.0 - init_weight)))
        nn.init.constant_(self.net[-1].bias, init_logit)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.net(features).squeeze(-1)
        gate = torch.sigmoid(logits)
        return gate, logits


class FEGX0Model(nn.Module):
    """Foundation-energy + internal formation-reference + residual FE head."""

    def __init__(
        self,
        *,
        element_mu: np.ndarray,
        element_bias: float,
        affine_alpha: float,
        affine_beta: np.ndarray,
        affine_bias: float,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        residual_mean: float,
        residual_std: float,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        trainable_reference: bool,
    ) -> None:
        super().__init__()
        self.element_reference = ElementReferenceLayer(element_mu, element_bias, trainable=trainable_reference)
        self.affine_reference = AffineReferenceLayer(affine_alpha, affine_beta, affine_bias, trainable=trainable_reference)
        self.residual_head = ResidualAdapter(MAX_Z + 5, hidden_dim, num_layers, dropout)
        self.register_buffer("feature_mean", torch.as_tensor(feature_mean, dtype=torch.float32))
        self.register_buffer("feature_std", torch.as_tensor(feature_std, dtype=torch.float32))
        self.register_buffer("residual_mean", torch.tensor(float(residual_mean), dtype=torch.float32))
        self.register_buffer("residual_std", torch.tensor(float(residual_std), dtype=torch.float32))

    def build_features(
        self,
        composition: torch.Tensor,
        foundation_energy: torch.Tensor,
        natoms: torch.Tensor,
        nelements: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        element_ref = self.element_reference(composition, foundation_energy)
        affine_ref = self.affine_reference(composition, foundation_energy)
        scalars = torch.stack(
            [
                foundation_energy,
                element_ref,
                affine_ref,
                torch.log1p(natoms),
                nelements,
            ],
            dim=1,
        )
        raw = torch.cat([composition, scalars], dim=1)
        features = (raw - self.feature_mean) / self.feature_std
        return features, element_ref, affine_ref

    def forward(
        self,
        composition: torch.Tensor,
        foundation_energy: torch.Tensor,
        natoms: torch.Tensor,
        nelements: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        features, element_ref, affine_ref = self.build_features(composition, foundation_energy, natoms, nelements)
        residual_norm = self.residual_head(features)
        delta = residual_norm * self.residual_std + self.residual_mean
        pred = element_ref + delta
        reference_gate = torch.ones_like(element_ref)
        return {
            "pred": pred,
            "adapter_delta": delta,
            "element_reference": element_ref,
            "dual_reference": element_ref,
            "aux_element_reference": element_ref,
            "dual_reference_gate": reference_gate,
            "affine_reference": affine_ref,
            "residual_norm": residual_norm,
        }


class FEGX1Model(nn.Module):
    """FE-GX-1 with chemistry pair pooling and structure/prototype tokens."""

    def __init__(
        self,
        *,
        element_mu: np.ndarray,
        element_bias: float,
        affine_alpha: float,
        affine_beta: np.ndarray,
        affine_bias: float,
        scalar_mean: np.ndarray,
        scalar_std: np.ndarray,
        target_mean: float,
        target_std: float,
        residual_mean: float,
        residual_std: float,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        trainable_reference: bool,
        pair_dim: int,
        spacegroup_dim: int,
        crystal_dim: int,
        structure_hidden_dim: int,
        crystal_vocab_size: int,
        residual_head_type: str = "mlp",
        num_experts: int = 3,
        router_hidden_dim: int = 64,
        router_temperature: float = 1.0,
        moe_router_init: str = "uniform",
        use_element_env: bool = False,
        element_env_dim: int = 16,
        element_env_hidden_dim: int = 64,
        element_env_delta_weight: float = 1.0,
        element_env_to_residual_head: bool = False,
        use_pair_env: bool = False,
        pair_env_dim: int = 16,
        pair_env_hidden_dim: int = 64,
        pair_env_delta_weight: float = 1.0,
        use_shell_pair_env: bool = False,
        shell_pair_env_dim: int = 16,
        shell_pair_env_hidden_dim: int = 64,
        shell_pair_env_delta_weight: float = 1.0,
        use_chem_pair_env: bool = False,
        chem_pair_env_dim: int = 16,
        chem_pair_env_hidden_dim: int = 64,
        chem_pair_env_delta_weight: float = 1.0,
        use_dual_foundation: bool = False,
        dual_foundation_dim: int = 16,
        dual_foundation_hidden_dim: int = 64,
        dual_foundation_mean: np.ndarray | None = None,
        dual_foundation_std: np.ndarray | None = None,
        use_risk_residual_branch: bool = False,
        risk_branch_hidden_dim: int = 64,
        risk_branch_gate_bias: float = -2.0,
        use_tail_features: bool = False,
        tail_feature_input_dim: int = 0,
        tail_feature_dim: int = 16,
        tail_feature_hidden_dim: int = 32,
        use_dual_reference_gate: bool = False,
        dual_reference_gate_hidden_dim: int = 16,
        dual_reference_gate_init_weight: float = 0.90,
        dual_reference_gate_use_raw_dual_stats: bool = False,
        use_primary_missing_branch: bool = False,
        primary_missing_branch_hidden_dim: int = 128,
        use_grouped_structure_adapter: bool = False,
        grouped_structure_group_dim: int = 16,
        grouped_structure_group_hidden_dim: int = 32,
        grouped_structure_context_hidden_dim: int = 64,
        grouped_structure_context_dim: int = 32,
        grouped_structure_interaction_dim: int = 32,
        grouped_structure_dropout: float = 0.05,
        grouped_structure_magnitude_bias: float = -1.5,
        anchor_mode: str = "fixed_anchor",
        learnable_anchor_init: float = 1.0,
    ) -> None:
        super().__init__()
        self.element_reference = ElementReferenceLayer(element_mu, element_bias, trainable=trainable_reference)
        self.affine_reference = AffineReferenceLayer(affine_alpha, affine_beta, affine_bias, trainable=trainable_reference)
        self.pair_embedding = nn.Embedding(MAX_Z * MAX_Z, pair_dim)
        self.spacegroup_embedding = nn.Embedding(232, spacegroup_dim)
        self.crystal_embedding = nn.Embedding(crystal_vocab_size, crystal_dim)
        self.register_buffer("scalar_mean", torch.as_tensor(scalar_mean, dtype=torch.float32))
        self.register_buffer("scalar_std", torch.as_tensor(scalar_std, dtype=torch.float32))
        self.register_buffer("target_mean", torch.tensor(float(target_mean), dtype=torch.float32))
        self.register_buffer("target_std", torch.tensor(float(target_std), dtype=torch.float32))
        self.register_buffer("residual_mean", torch.tensor(float(residual_mean), dtype=torch.float32))
        self.register_buffer("residual_std", torch.tensor(float(residual_std), dtype=torch.float32))
        if dual_foundation_mean is None:
            dual_foundation_mean = np.zeros(len(DUAL_FOUNDATION_STATS), dtype=np.float32)
        if dual_foundation_std is None:
            dual_foundation_std = np.ones(len(DUAL_FOUNDATION_STATS), dtype=np.float32)
        self.register_buffer("dual_foundation_mean", torch.as_tensor(dual_foundation_mean, dtype=torch.float32))
        self.register_buffer("dual_foundation_std", torch.as_tensor(dual_foundation_std, dtype=torch.float32))
        self.element_env_delta_weight = float(element_env_delta_weight)
        self.pair_env_delta_weight = float(pair_env_delta_weight)
        self.shell_pair_env_delta_weight = float(shell_pair_env_delta_weight)
        self.chem_pair_env_delta_weight = float(chem_pair_env_delta_weight)
        self.element_env_to_residual_head = bool(use_element_env and element_env_to_residual_head)
        self.use_dual_reference_gate = bool(use_dual_reference_gate)
        self.dual_reference_gate_use_raw_dual_stats = bool(dual_reference_gate_use_raw_dual_stats)
        self.use_primary_missing_branch = bool(use_primary_missing_branch)
        self.use_grouped_structure_adapter = bool(use_grouped_structure_adapter)
        self.anchor_mode = str(anchor_mode).lower()
        valid_anchor_modes = {"fixed_anchor", "learnable_anchor", "stacking", "target_only"}
        if self.anchor_mode not in valid_anchor_modes:
            raise ValueError(
                f"Unsupported anchor_mode={anchor_mode!r}; expected one of {sorted(valid_anchor_modes)}"
            )
        self.grouped_structure_group_names = list(GROUPED_STRUCTURE_GROUP_NAMES)
        self.structure_scalar_index = {name: idx for idx, name in enumerate(STRUCTURE_SCALAR_COLS)}
        if self.anchor_mode == "learnable_anchor":
            self.anchor_alpha = nn.Parameter(torch.tensor(float(learnable_anchor_init), dtype=torch.float32))
        else:
            anchor_alpha_value = 1.0 if self.anchor_mode == "fixed_anchor" else float(learnable_anchor_init)
            self.register_buffer("anchor_alpha_buffer", torch.tensor(anchor_alpha_value, dtype=torch.float32))
        self.element_env_encoder = (
            ElementEnvEncoder(
                stats_dim=len(ELEMENT_ENV_STATS),
                env_dim=element_env_dim,
                hidden_dim=element_env_hidden_dim,
                dropout=dropout,
            )
            if use_element_env
            else None
        )
        self.pair_env_encoder = (
            PairEnvEncoder(
                stats_dim=len(PAIR_ENV_STATS),
                pair_dim=pair_env_dim,
                hidden_dim=pair_env_hidden_dim,
                dropout=dropout,
            )
            if use_pair_env
            else None
        )
        self.shell_pair_env_encoder = (
            ShellPairEnvEncoder(
                stats_dim=len(SHELL_PAIR_ENV_STATS),
                pair_dim=shell_pair_env_dim,
                hidden_dim=shell_pair_env_hidden_dim,
                dropout=dropout,
            )
            if use_shell_pair_env
            else None
        )
        self.chem_pair_env_encoder = (
            ChemPairEnvEncoder(
                stats_dim=len(CHEM_PAIR_ENV_STATS),
                pair_dim=chem_pair_env_dim,
                hidden_dim=chem_pair_env_hidden_dim,
                dropout=dropout,
            )
            if use_chem_pair_env
            else None
        )
        self.dual_foundation_encoder = (
            DualFoundationEncoder(
                stats_dim=len(DUAL_FOUNDATION_STATS),
                token_dim=dual_foundation_dim,
                hidden_dim=dual_foundation_hidden_dim,
                dropout=dropout,
            )
            if use_dual_foundation
            else None
        )

        scalar_dim = len(scalar_mean)
        structure_in = scalar_dim + spacegroup_dim + crystal_dim
        self.structure_encoder = nn.Sequential(
            nn.Linear(structure_in, structure_hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(structure_hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(structure_hidden_dim, structure_hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(structure_hidden_dim),
        )
        head_in = MAX_Z + 5 + pair_dim + structure_hidden_dim
        if self.dual_foundation_encoder is not None:
            head_in += dual_foundation_dim
        if self.element_env_to_residual_head:
            head_in += element_env_dim
        self.tail_feature_encoder = None
        if use_tail_features:
            if tail_feature_input_dim <= 0:
                raise ValueError("use_tail_features=True requires tail_feature_input_dim > 0")
            token_dim = max(1, int(tail_feature_dim))
            hidden_tail = max(token_dim, int(tail_feature_hidden_dim))
            self.tail_feature_encoder = nn.Sequential(
                nn.Linear(int(tail_feature_input_dim), hidden_tail),
                nn.SiLU(),
                nn.LayerNorm(hidden_tail),
                nn.Dropout(dropout),
                nn.Linear(hidden_tail, token_dim),
                nn.SiLU(),
                nn.LayerNorm(token_dim),
            )
            head_in += token_dim
        self.grouped_structure_adapter = (
            GroupedStructureAdapter(
                pair_dim=pair_dim,
                spacegroup_dim=spacegroup_dim,
                crystal_dim=crystal_dim,
                dual_stats_dim=len(DUAL_FOUNDATION_STATS),
                group_dim=grouped_structure_group_dim,
                group_hidden_dim=grouped_structure_group_hidden_dim,
                context_hidden_dim=grouped_structure_context_hidden_dim,
                context_dim=grouped_structure_context_dim,
                interaction_dim=grouped_structure_interaction_dim,
                dropout=grouped_structure_dropout,
                magnitude_bias=grouped_structure_magnitude_bias,
            )
            if self.use_grouped_structure_adapter
            else None
        )
        gate_in = head_in + (len(DUAL_FOUNDATION_STATS) if self.dual_reference_gate_use_raw_dual_stats else 0)
        self.dual_reference_gate = (
            DualReferenceGate(
                in_dim=gate_in,
                hidden_dim=dual_reference_gate_hidden_dim,
                init_weight=dual_reference_gate_init_weight,
                dropout=dropout,
            )
            if self.use_dual_reference_gate
            else None
        )
        self.residual_head_type = str(residual_head_type).lower()
        if self.residual_head_type == "moe":
            self.residual_head = ResidualMoEHead(
                head_in,
                hidden_dim,
                num_layers,
                dropout,
                num_experts=num_experts,
                router_hidden_dim=router_hidden_dim,
                router_temperature=router_temperature,
                router_init=moe_router_init,
            )
        else:
            self.residual_head = ResidualAdapter(head_in, hidden_dim, num_layers, dropout)
        self.no_primary_head = None
        if self.use_primary_missing_branch:
            missing_hidden_dim = max(32, int(primary_missing_branch_hidden_dim))
            self.no_primary_head = nn.Sequential(
                nn.Linear(head_in + 1, missing_hidden_dim),
                nn.SiLU(),
                nn.LayerNorm(missing_hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(missing_hidden_dim, missing_hidden_dim),
                nn.SiLU(),
                nn.LayerNorm(missing_hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(missing_hidden_dim, 1),
            )
            nn.init.zeros_(self.no_primary_head[-1].weight)
            nn.init.zeros_(self.no_primary_head[-1].bias)
        self.risk_head = nn.Sequential(
            nn.Linear(head_in, max(32, hidden_dim // 2)),
            nn.SiLU(),
            nn.LayerNorm(max(32, hidden_dim // 2)),
            nn.Dropout(dropout),
            nn.Linear(max(32, hidden_dim // 2), 1),
        )
        self.positive_head = nn.Sequential(
            nn.Linear(head_in, max(32, hidden_dim // 2)),
            nn.SiLU(),
            nn.LayerNorm(max(32, hidden_dim // 2)),
            nn.Dropout(dropout),
            nn.Linear(max(32, hidden_dim // 2), 1),
        )
        self.near_zero_head = nn.Sequential(
            nn.Linear(head_in, max(32, hidden_dim // 2)),
            nn.SiLU(),
            nn.LayerNorm(max(32, hidden_dim // 2)),
            nn.Dropout(dropout),
            nn.Linear(max(32, hidden_dim // 2), 1),
        )
        self.risk_residual_gate = None
        self.risk_residual_head = None
        if use_risk_residual_branch:
            branch_dim = max(8, int(risk_branch_hidden_dim))
            self.risk_residual_gate = nn.Sequential(
                nn.Linear(head_in, branch_dim),
                nn.SiLU(),
                nn.LayerNorm(branch_dim),
                nn.Dropout(dropout),
                nn.Linear(branch_dim, 1),
            )
            self.risk_residual_head = nn.Sequential(
                nn.Linear(head_in, branch_dim),
                nn.SiLU(),
                nn.LayerNorm(branch_dim),
                nn.Dropout(dropout),
                nn.Linear(branch_dim, 1),
            )
            nn.init.zeros_(self.risk_residual_gate[-1].weight)
            nn.init.constant_(self.risk_residual_gate[-1].bias, float(risk_branch_gate_bias))
            nn.init.zeros_(self.risk_residual_head[-1].weight)
            nn.init.zeros_(self.risk_residual_head[-1].bias)

    def pair_pool(self, composition: torch.Tensor) -> torch.Tensor:
        weights = composition[:, :, None] * composition[:, None, :]
        index = torch.arange(MAX_Z, device=composition.device, dtype=torch.long)
        pair_ids = (index[:, None] * MAX_Z + index[None, :]).reshape(-1)
        pair_emb = self.pair_embedding(pair_ids).reshape(MAX_Z, MAX_Z, -1)
        return torch.einsum("bij,ijd->bd", weights, pair_emb)

    def _anchor_alpha(self) -> torch.Tensor | None:
        if self.anchor_mode in {"fixed_anchor", "learnable_anchor"}:
            if hasattr(self, "anchor_alpha"):
                return self.anchor_alpha
            return self.anchor_alpha_buffer
        return None

    def build_features(
        self,
        composition: torch.Tensor,
        foundation_energy: torch.Tensor,
        natoms: torch.Tensor,
        nelements: torch.Tensor,
        structure_scalars: torch.Tensor,
        spacegroup_index: torch.Tensor,
        crystal_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        element_ref = self.element_reference(composition, foundation_energy)
        affine_ref = self.affine_reference(composition, foundation_energy)
        source_foundation = foundation_energy
        source_element_ref = element_ref
        source_affine_ref = affine_ref
        if self.anchor_mode == "target_only":
            source_foundation = torch.zeros_like(source_foundation)
            source_element_ref = torch.zeros_like(source_element_ref)
            source_affine_ref = torch.zeros_like(source_affine_ref)
        base_scalars = torch.stack(
            [
                source_foundation,
                source_element_ref,
                source_affine_ref,
                torch.log1p(natoms),
                nelements,
            ],
            dim=1,
        )
        pair = self.pair_pool(composition)
        struct_norm = (structure_scalars - self.scalar_mean) / self.scalar_std
        sg_emb = self.spacegroup_embedding(spacegroup_index.clamp(0, 231))
        crystal_emb = self.crystal_embedding(crystal_index)
        struct = self.structure_encoder(torch.cat([struct_norm, sg_emb, crystal_emb], dim=1))
        features = torch.cat([composition, base_scalars, pair, struct], dim=1)
        return features, element_ref, affine_ref, pair

    def _structure_scalar(self, structure_scalars: torch.Tensor, name: str) -> torch.Tensor:
        return structure_scalars[:, self.structure_scalar_index[name]]

    def _build_grouped_structure_inputs(
        self,
        *,
        foundation_energy: torch.Tensor,
        natoms: torch.Tensor,
        nelements: torch.Tensor,
        structure_scalars: torch.Tensor,
        spacegroup_index: torch.Tensor,
        crystal_index: torch.Tensor,
        pair: torch.Tensor,
        element_ref: torch.Tensor,
        affine_ref: torch.Tensor,
        dual_foundation_stats: torch.Tensor | None,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        eps = 1e-6
        volume = self._structure_scalar(structure_scalars, "volume_per_atom").clamp_min(eps)
        density = self._structure_scalar(structure_scalars, "density").clamp_min(eps)
        ineq = self._structure_scalar(structure_scalars, "symmetry_inequivalent_sites").clamp_min(0.0)
        coord_mean = self._structure_scalar(structure_scalars, "coord_mean").clamp_min(eps)
        coord_std = self._structure_scalar(structure_scalars, "coord_std").clamp_min(0.0)
        coord_min = self._structure_scalar(structure_scalars, "coord_min").clamp_min(0.0)
        coord_max = self._structure_scalar(structure_scalars, "coord_max").clamp_min(0.0)
        bond_mean = self._structure_scalar(structure_scalars, "bond_dist_mean").clamp_min(eps)
        bond_std = self._structure_scalar(structure_scalars, "bond_dist_std").clamp_min(0.0)
        bond_min = self._structure_scalar(structure_scalars, "bond_dist_min").clamp_min(0.0)
        lattice_a = self._structure_scalar(structure_scalars, "lattice_a").clamp_min(eps)
        lattice_b = self._structure_scalar(structure_scalars, "lattice_b").clamp_min(eps)
        lattice_c = self._structure_scalar(structure_scalars, "lattice_c").clamp_min(eps)
        lattice_alpha = self._structure_scalar(structure_scalars, "lattice_alpha")
        lattice_beta = self._structure_scalar(structure_scalars, "lattice_beta")
        lattice_gamma = self._structure_scalar(structure_scalars, "lattice_gamma")

        abc = torch.stack([lattice_a, lattice_b, lattice_c], dim=1)
        abc_mean = abc.mean(dim=1).clamp_min(eps)
        abc_std = abc.std(dim=1, unbiased=False)
        angle_scale = float(np.pi / 180.0)
        sg_emb = self.spacegroup_embedding(spacegroup_index.clamp(0, 231))
        crystal_emb = self.crystal_embedding(crystal_index)
        natoms_safe = natoms.clamp_min(1.0)
        dual_context = torch.zeros(
            (structure_scalars.shape[0], len(DUAL_FOUNDATION_STATS)),
            device=structure_scalars.device,
            dtype=structure_scalars.dtype,
        )
        if dual_foundation_stats is not None and dual_foundation_stats.numel() > 0:
            dual_context = dual_foundation_stats.to(device=structure_scalars.device, dtype=structure_scalars.dtype)

        groups = {
            "scale": torch.stack([torch.log(volume), torch.log(density)], dim=1),
            "shape": torch.stack(
                [
                    torch.log(lattice_b / lattice_a),
                    torch.log(lattice_c / lattice_a),
                    abc_std / abc_mean,
                    torch.cos(lattice_alpha * angle_scale),
                    torch.cos(lattice_beta * angle_scale),
                    torch.cos(lattice_gamma * angle_scale),
                ],
                dim=1,
            ),
            "symmetry": torch.cat(
                [
                    sg_emb,
                    crystal_emb,
                    torch.log1p(ineq).unsqueeze(1),
                    (ineq / natoms_safe).unsqueeze(1),
                ],
                dim=1,
            ),
            "coordination": torch.stack(
                [
                    coord_mean,
                    coord_std / coord_mean,
                    coord_min / coord_mean,
                    coord_max / coord_mean,
                    (coord_max - coord_min) / coord_mean,
                ],
                dim=1,
            ),
            "bond": torch.stack(
                [
                    torch.log(bond_mean),
                    bond_std / bond_mean,
                    bond_min / bond_mean,
                    (bond_mean - bond_min) / bond_mean,
                ],
                dim=1,
            ),
        }
        context_foundation = foundation_energy
        context_element_ref = element_ref
        context_affine_ref = affine_ref
        if self.anchor_mode == "target_only":
            context_foundation = torch.zeros_like(context_foundation)
            context_element_ref = torch.zeros_like(context_element_ref)
            context_affine_ref = torch.zeros_like(context_affine_ref)
        context = torch.cat(
            [
                context_foundation.unsqueeze(1),
                context_element_ref.unsqueeze(1),
                context_affine_ref.unsqueeze(1),
                torch.log1p(natoms.clamp_min(0.0)).unsqueeze(1),
                nelements.unsqueeze(1),
                pair,
                dual_context,
            ],
            dim=1,
        )
        return groups, context

    def forward(
        self,
        composition: torch.Tensor,
        foundation_energy: torch.Tensor,
        natoms: torch.Tensor,
        nelements: torch.Tensor,
        structure_scalars: torch.Tensor,
        spacegroup_index: torch.Tensor,
        crystal_index: torch.Tensor,
        element_env_stats: torch.Tensor | None = None,
        pair_env_ids: torch.Tensor | None = None,
        pair_env_stats: torch.Tensor | None = None,
        shell_pair_env_ids: torch.Tensor | None = None,
        shell_pair_env_stats: torch.Tensor | None = None,
        chem_pair_env_ids: torch.Tensor | None = None,
        chem_pair_env_stats: torch.Tensor | None = None,
        dual_foundation_stats: torch.Tensor | None = None,
        tail_features: torch.Tensor | None = None,
        primary_missing: torch.Tensor | None = None,
        return_features: bool = False,
    ) -> dict[str, torch.Tensor]:
        features, element_ref, affine_ref, pair = self.build_features(
            composition,
            foundation_energy,
            natoms,
            nelements,
            structure_scalars,
            spacegroup_index,
            crystal_index,
        )
        env_delta_norm = torch.zeros((features.shape[0],), device=features.device, dtype=features.dtype)
        env_embedding = torch.empty((features.shape[0], 0), device=features.device)
        if self.element_env_encoder is not None and element_env_stats is not None and element_env_stats.numel() > 0:
            env_delta_norm, env_embedding = self.element_env_encoder(element_env_stats)
        pair_env_delta_norm = torch.zeros((features.shape[0],), device=features.device, dtype=features.dtype)
        pair_env_embedding = torch.empty((features.shape[0], 0), device=features.device)
        if (
            self.pair_env_encoder is not None
            and pair_env_ids is not None
            and pair_env_stats is not None
            and pair_env_ids.numel() > 0
            and pair_env_stats.numel() > 0
        ):
            pair_env_delta_norm, pair_env_embedding = self.pair_env_encoder(pair_env_ids, pair_env_stats)
        shell_pair_env_delta_norm = torch.zeros((features.shape[0],), device=features.device, dtype=features.dtype)
        shell_pair_env_embedding = torch.empty((features.shape[0], 0), device=features.device)
        if (
            self.shell_pair_env_encoder is not None
            and shell_pair_env_ids is not None
            and shell_pair_env_stats is not None
            and shell_pair_env_ids.numel() > 0
            and shell_pair_env_stats.numel() > 0
        ):
            shell_pair_env_delta_norm, shell_pair_env_embedding = self.shell_pair_env_encoder(
                shell_pair_env_ids, shell_pair_env_stats
            )
        chem_pair_env_delta_norm = torch.zeros((features.shape[0],), device=features.device, dtype=features.dtype)
        chem_pair_env_embedding = torch.empty((features.shape[0], 0), device=features.device)
        if (
            self.chem_pair_env_encoder is not None
            and chem_pair_env_ids is not None
            and chem_pair_env_stats is not None
            and chem_pair_env_ids.numel() > 0
            and chem_pair_env_stats.numel() > 0
        ):
            chem_pair_env_delta_norm, chem_pair_env_embedding = self.chem_pair_env_encoder(
                chem_pair_env_ids, chem_pair_env_stats
            )
        residual_features = features
        dual_foundation_embedding = torch.empty((features.shape[0], 0), device=features.device)
        if self.dual_foundation_encoder is not None:
            if dual_foundation_stats is None or dual_foundation_stats.numel() == 0:
                dual_dim = self.dual_foundation_encoder.encoder[-3].out_features
                dual_foundation_embedding = torch.zeros((features.shape[0], dual_dim), device=features.device, dtype=features.dtype)
            else:
                dual_foundation_embedding = self.dual_foundation_encoder(dual_foundation_stats)
            residual_features = torch.cat([residual_features, dual_foundation_embedding], dim=1)
        if self.element_env_to_residual_head:
            if env_embedding.numel() == 0:
                env_dim = self.element_env_encoder.element_embedding.embedding_dim if self.element_env_encoder is not None else 0
                env_embedding = torch.zeros((features.shape[0], env_dim), device=features.device, dtype=features.dtype)
            residual_features = torch.cat([residual_features, env_embedding], dim=1)
        tail_feature_embedding = torch.empty((features.shape[0], 0), device=features.device)
        if self.tail_feature_encoder is not None:
            if tail_features is None or tail_features.numel() == 0:
                tail_in = self.tail_feature_encoder[0].in_features
                tail_features = torch.zeros((features.shape[0], tail_in), device=features.device, dtype=features.dtype)
            tail_feature_embedding = self.tail_feature_encoder(tail_features)
            residual_features = torch.cat([residual_features, tail_feature_embedding], dim=1)
        if primary_missing is None:
            primary_missing = torch.zeros((features.shape[0],), device=features.device, dtype=features.dtype)
        else:
            primary_missing = primary_missing.to(device=features.device, dtype=features.dtype).reshape(-1)
        reference_gate = torch.ones((features.shape[0],), device=features.device, dtype=features.dtype)
        reference_gate_logit = torch.zeros((features.shape[0],), device=features.device, dtype=features.dtype)
        dual_reference = element_ref
        aux_element_reference = torch.zeros((features.shape[0],), device=features.device, dtype=features.dtype)
        if self.dual_reference_gate is not None:
            if dual_foundation_stats is None or dual_foundation_stats.numel() == 0:
                aux_element_reference = element_ref
                gate_features = residual_features
                if self.dual_reference_gate_use_raw_dual_stats:
                    zeros = torch.zeros(
                        (features.shape[0], len(DUAL_FOUNDATION_STATS)),
                        device=features.device,
                        dtype=features.dtype,
                    )
                    gate_features = torch.cat([gate_features, zeros], dim=1)
            else:
                aux_idx = DUAL_FOUNDATION_STATS.index("aux_element_reference")
                aux_element_reference = dual_foundation_stats[:, aux_idx] * self.dual_foundation_std[aux_idx] + self.dual_foundation_mean[aux_idx]
                gate_features = residual_features
                if self.dual_reference_gate_use_raw_dual_stats:
                    gate_features = torch.cat([gate_features, dual_foundation_stats], dim=1)
            reference_gate, reference_gate_logit = self.dual_reference_gate(gate_features)
            dual_reference = reference_gate * element_ref + (1.0 - reference_gate) * aux_element_reference
        router_weights = torch.empty((features.shape[0], 0), device=features.device)
        expert_outputs = torch.empty((features.shape[0], 0), device=features.device)
        if self.residual_head_type == "moe":
            residual_norm, router_weights, expert_outputs = self.residual_head(residual_features)
        else:
            residual_norm = self.residual_head(residual_features)
        risk_branch_gate_logit = torch.zeros((features.shape[0],), device=features.device, dtype=features.dtype)
        risk_branch_gate = torch.zeros((features.shape[0],), device=features.device, dtype=features.dtype)
        risk_branch_delta_norm = torch.zeros((features.shape[0],), device=features.device, dtype=features.dtype)
        if self.risk_residual_gate is not None and self.risk_residual_head is not None:
            risk_branch_gate_logit = self.risk_residual_gate(residual_features).squeeze(-1)
            risk_branch_gate = torch.sigmoid(risk_branch_gate_logit)
            risk_branch_delta_norm = risk_branch_gate * self.risk_residual_head(residual_features).squeeze(-1)
        total_residual_norm = (
            residual_norm
            + self.element_env_delta_weight * env_delta_norm
            + self.pair_env_delta_weight * pair_env_delta_norm
            + self.shell_pair_env_delta_weight * shell_pair_env_delta_norm
            + self.chem_pair_env_delta_weight * chem_pair_env_delta_norm
            + risk_branch_delta_norm
        )
        direct_target_mode = self.anchor_mode in {"stacking", "target_only"}
        branch_scale = self.target_std if direct_target_mode else self.residual_std
        if direct_target_mode:
            delta = total_residual_norm * self.target_std + self.target_mean
            anchor_component = torch.zeros((features.shape[0],), device=features.device, dtype=features.dtype)
            no_primary_pred = delta
        else:
            delta = total_residual_norm * self.residual_std + self.residual_mean
            anchor_alpha = self._anchor_alpha()
            if anchor_alpha is None:
                raise RuntimeError(f"anchor_mode={self.anchor_mode!r} expected a reference anchor weight")
            anchor_component = anchor_alpha * dual_reference
            no_primary_pred = anchor_component + delta
        if self.no_primary_head is not None:
            no_primary_features = residual_features.clone()
            no_primary_features[:, MAX_Z : MAX_Z + 3] = 0.0
            no_primary_input = torch.cat([no_primary_features, primary_missing.unsqueeze(1)], dim=1)
            no_primary_norm = self.no_primary_head(no_primary_input).squeeze(-1)
            no_primary_pred = no_primary_norm * self.target_std + self.target_mean
        primary_pred = delta if direct_target_mode else anchor_component + delta
        base_pred = torch.where(primary_missing > 0.5, no_primary_pred, primary_pred)
        adapter_delta = base_pred - anchor_component
        grouped_structure_delta = torch.zeros((features.shape[0],), device=features.device, dtype=features.dtype)
        grouped_structure_raw_delta = torch.zeros((features.shape[0],), device=features.device, dtype=features.dtype)
        grouped_structure_magnitude = torch.zeros((features.shape[0],), device=features.device, dtype=features.dtype)
        grouped_structure_weights = torch.zeros(
            (features.shape[0], len(self.grouped_structure_group_names)),
            device=features.device,
            dtype=features.dtype,
        )
        if self.grouped_structure_adapter is not None:
            grouped_inputs, grouped_context = self._build_grouped_structure_inputs(
                foundation_energy=foundation_energy,
                natoms=natoms,
                nelements=nelements,
                structure_scalars=structure_scalars,
                spacegroup_index=spacegroup_index,
                crystal_index=crystal_index,
                pair=pair,
                element_ref=element_ref,
                affine_ref=affine_ref,
                dual_foundation_stats=dual_foundation_stats,
            )
            grouped_output = self.grouped_structure_adapter(grouped_inputs, grouped_context)
            grouped_structure_delta = grouped_output["delta"]
            grouped_structure_raw_delta = grouped_output["raw_delta"]
            grouped_structure_magnitude = grouped_output["magnitude"]
            grouped_structure_weights = grouped_output["weights"]
        pred = base_pred + grouped_structure_delta
        result = {
            "pred": pred,
            "base_pred": base_pred,
            "no_primary_pred": no_primary_pred,
            "adapter_delta": adapter_delta,
            "anchor_component": anchor_component,
            "grouped_structure_delta": grouped_structure_delta,
            "grouped_structure_raw_delta": grouped_structure_raw_delta,
            "grouped_structure_magnitude": grouped_structure_magnitude,
            "grouped_structure_weights": grouped_structure_weights,
            "moe_delta": residual_norm * branch_scale,
            "env_delta": self.element_env_delta_weight * env_delta_norm * branch_scale,
            "pair_env_delta": self.pair_env_delta_weight * pair_env_delta_norm * branch_scale,
            "shell_pair_env_delta": self.shell_pair_env_delta_weight * shell_pair_env_delta_norm * branch_scale,
            "chem_pair_env_delta": self.chem_pair_env_delta_weight * chem_pair_env_delta_norm * branch_scale,
            "risk_residual_delta": risk_branch_delta_norm * branch_scale,
            "element_reference": element_ref,
            "dual_reference": dual_reference,
            "aux_element_reference": aux_element_reference,
            "dual_reference_gate": reference_gate,
            "dual_reference_gate_logit": reference_gate_logit,
            "affine_reference": affine_ref,
            "anchor_alpha": (
                self._anchor_alpha().expand(features.shape[0])
                if self._anchor_alpha() is not None
                else torch.empty((features.shape[0], 0), device=features.device)
            ),
            "primary_missing": primary_missing,
            "residual_norm": total_residual_norm,
            "moe_residual_norm": residual_norm,
            "env_residual_norm": env_delta_norm,
            "pair_env_residual_norm": pair_env_delta_norm,
            "shell_pair_env_residual_norm": shell_pair_env_delta_norm,
            "chem_pair_env_residual_norm": chem_pair_env_delta_norm,
            "risk_branch_residual_norm": risk_branch_delta_norm,
            "risk_branch_gate": risk_branch_gate,
            "risk_branch_gate_logit": risk_branch_gate_logit,
            "log_sigma": self.risk_head(residual_features).squeeze(-1),
            "positive_logit": self.positive_head(residual_features).squeeze(-1),
            "near_zero_logit": self.near_zero_head(residual_features).squeeze(-1),
            "router_weights": router_weights,
            "expert_outputs": expert_outputs,
            "env_embedding": env_embedding,
            "pair_env_embedding": pair_env_embedding,
            "shell_pair_env_embedding": shell_pair_env_embedding,
            "chem_pair_env_embedding": chem_pair_env_embedding,
            "dual_foundation_embedding": dual_foundation_embedding,
            "tail_feature_embedding": tail_feature_embedding,
        }
        if return_features:
            result["features"] = residual_features
        return result


def load_reference_coefficients(path: Path) -> dict[str, float | np.ndarray]:
    coeffs = pd.read_csv(path)
    required = {"calibration", "coefficient", "value"}
    if not required.issubset(coeffs.columns):
        raise KeyError(f"{path} must contain columns {sorted(required)}")

    def values_for(name: str) -> dict[str, float]:
        group = coeffs[coeffs["calibration"] == name]
        return {str(row.coefficient): float(row.value) for row in group.itertuples(index=False)}

    element = values_for("element_reference")
    affine = values_for("affine_energy_composition")
    if not element or not affine:
        raise RuntimeError(f"{path} is missing element_reference or affine_energy_composition coefficients")
    mu = np.asarray([element.get(f"mu_Z{z}", 0.0) for z in range(1, MAX_Z + 1)], dtype=np.float32)
    beta = np.asarray([affine.get(f"beta_Z{z}", 0.0) for z in range(1, MAX_Z + 1)], dtype=np.float32)
    return {
        "element_mu": mu,
        "element_bias": float(element.get("bias", 0.0)),
        "affine_alpha": float(affine.get("alpha_energy", 0.0)),
        "affine_beta": beta,
        "affine_bias": float(affine.get("bias", 0.0)),
    }


def fegx0_raw_features(
    *,
    composition: np.ndarray,
    foundation_energy: np.ndarray,
    natoms: np.ndarray,
    nelements: np.ndarray,
    element_reference: np.ndarray,
    affine_reference: np.ndarray,
) -> np.ndarray:
    scalars = np.column_stack(
        [
            foundation_energy,
            element_reference,
            affine_reference,
            np.log1p(natoms),
            nelements,
        ]
    ).astype(np.float32)
    return np.concatenate([composition.astype(np.float32), scalars], axis=1)


def load_fegx0_frame(calibrated_predictions: Path, feature_cache: Path) -> pd.DataFrame:
    frame = load_adapter_frame(calibrated_predictions, feature_cache)
    needed = ["split", "id", "cif_path", "y_true", "model_energy_per_atom", "natoms", "nelements", *X_COLS]
    missing = [col for col in needed if col not in frame.columns]
    if missing:
        raise KeyError(f"Missing FE-GX-0 input columns: {missing[:10]}")
    return frame


STRUCTURE_SCALAR_COLS = [
    "volume_per_atom",
    "density",
    "symmetry_inequivalent_sites",
    "coord_mean",
    "coord_std",
    "coord_min",
    "coord_max",
    "bond_dist_mean",
    "bond_dist_std",
    "bond_dist_min",
    "lattice_a",
    "lattice_b",
    "lattice_c",
    "lattice_alpha",
    "lattice_beta",
    "lattice_gamma",
]

CRYSTAL_SYSTEMS = ["unknown", "triclinic", "monoclinic", "orthorhombic", "tetragonal", "trigonal", "hexagonal", "cubic"]


def load_fegx1_frame(calibrated_predictions: Path, feature_cache: Path, structure_cache: Path) -> pd.DataFrame:
    frame = load_fegx0_frame(calibrated_predictions, feature_cache)
    metadata_cols = ["id", "cif_path", "spacegroup_number", "crystal_system", *STRUCTURE_SCALAR_COLS]
    metadata = pd.read_csv(structure_cache, keep_default_na=False, usecols=lambda col: col in metadata_cols)
    frame = frame.merge(metadata, on=["id", "cif_path"], how="left")
    missing = [col for col in ["spacegroup_number", "crystal_system", *STRUCTURE_SCALAR_COLS] if col not in frame.columns]
    if missing:
        raise KeyError(f"Missing FE-GX-1 structure columns: {missing}")
    return frame


def element_env_columns() -> list[str]:
    return [f"env_Z{z}_{stat}" for z in range(1, MAX_Z + 1) for stat in ELEMENT_ENV_STATS]


def load_element_env_features(frame: pd.DataFrame, element_env_cache: Path, train_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    cols = ["id", "cif_path", *element_env_columns()]
    env = pd.read_csv(element_env_cache, keep_default_na=False, usecols=lambda col: col in cols)
    frame_keys = frame[["id", "cif_path"]].copy()
    merged = frame_keys.merge(env, on=["id", "cif_path"], how="left")
    missing_cols = [col for col in element_env_columns() if col not in merged.columns]
    if missing_cols:
        raise KeyError(f"Missing element-env columns: {missing_cols[:5]}")
    values = merged[element_env_columns()].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    missing_rows = np.isnan(values).any(axis=1)
    values[~np.isfinite(values)] = 0.0
    values = values.reshape(len(frame), MAX_Z, len(ELEMENT_ENV_STATS))
    mean = values[train_mask].reshape(-1, len(ELEMENT_ENV_STATS)).mean(axis=0).astype(np.float32)
    std = values[train_mask].reshape(-1, len(ELEMENT_ENV_STATS)).std(axis=0).astype(np.float32)
    mean[0] = 0.0
    mean[1] = 0.0
    std[std < 1e-8] = 1.0
    std[0] = 1.0
    std[1] = 1.0
    values_norm = values.copy()
    for idx in range(2, len(ELEMENT_ENV_STATS)):
        values_norm[:, :, idx] = (values[:, :, idx] - mean[idx]) / std[idx]
    summary = {
        "cache": str(element_env_cache),
        "stats": ELEMENT_ENV_STATS,
        "missing_rows": int(missing_rows.sum()),
        "shape": list(values_norm.shape),
    }
    return values_norm.astype(np.float32), mean, std, summary


def load_pair_env_features(
    frame: pd.DataFrame,
    pair_env_cache: Path,
    train_mask: np.ndarray,
    *,
    max_pairs: int,
    ablation: str = "none",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    cols = ["id", "cif_path", "pair_id", *PAIR_ENV_STATS]
    env = pd.read_csv(pair_env_cache, keep_default_na=False, usecols=lambda col: col in cols)
    env["id"] = env["id"].astype(str)
    env["cif_path"] = env["cif_path"].astype(str)
    env["pair_id"] = pd.to_numeric(env["pair_id"], errors="coerce").fillna(0).astype(np.int64)
    for stat in PAIR_ENV_STATS:
        env[stat] = pd.to_numeric(env[stat], errors="coerce").fillna(0.0).astype(np.float32)
    env = env.sort_values(["id", "cif_path", "pair_frac", "bond_count_frac"], ascending=[True, True, False, False])

    row_lookup = {
        (str(row.id), str(row.cif_path)): idx
        for idx, row in enumerate(frame[["id", "cif_path"]].itertuples(index=False))
    }
    max_pairs = int(max(1, max_pairs))
    ablation = str(ablation).lower()
    valid_ablations = {"none", "no_geometry", "no_identity", "no_same_pair", "no_bond_count"}
    if ablation not in valid_ablations:
        raise ValueError(f"Unknown pair_env_ablation={ablation!r}; expected one of {sorted(valid_ablations)}")

    pair_ids = np.zeros((len(frame), max_pairs), dtype=np.int64)
    pair_stats = np.zeros((len(frame), max_pairs, len(PAIR_ENV_STATS)), dtype=np.float32)
    counts = np.zeros(len(frame), dtype=np.int32)
    overflow = 0
    for key, group in env.groupby(["id", "cif_path"], sort=False):
        row_idx = row_lookup.get((str(key[0]), str(key[1])))
        if row_idx is None:
            continue
        limited = group.head(max_pairs)
        n_pairs = len(limited)
        counts[row_idx] = n_pairs
        if len(group) > max_pairs:
            overflow += 1
        if n_pairs:
            pair_ids[row_idx, :n_pairs] = limited["pair_id"].to_numpy(dtype=np.int64)
            pair_stats[row_idx, :n_pairs, :] = limited[PAIR_ENV_STATS].to_numpy(dtype=np.float32)

    mean = pair_stats[train_mask].reshape(-1, len(PAIR_ENV_STATS)).mean(axis=0).astype(np.float32)
    std = pair_stats[train_mask].reshape(-1, len(PAIR_ENV_STATS)).std(axis=0).astype(np.float32)
    mean[0] = 0.0
    mean[1] = 0.0
    mean[-1] = 0.0
    std[std < 1e-8] = 1.0
    std[0] = 1.0
    std[1] = 1.0
    std[-1] = 1.0
    stats_norm = pair_stats.copy()
    for idx in range(2, len(PAIR_ENV_STATS) - 1):
        stats_norm[:, :, idx] = (pair_stats[:, :, idx] - mean[idx]) / std[idx]
    if ablation == "no_geometry":
        for stat in ["bond_mean", "bond_std", "bond_min", "bond_max"]:
            stats_norm[:, :, PAIR_ENV_STATS.index(stat)] = 0.0
    elif ablation == "no_identity":
        pair_ids.fill(0)
    elif ablation == "no_same_pair":
        stats_norm[:, :, PAIR_ENV_STATS.index("same_pair")] = 0.0
    elif ablation == "no_bond_count":
        stats_norm[:, :, PAIR_ENV_STATS.index("bond_count_frac")] = 0.0
    summary = {
        "cache": str(pair_env_cache),
        "stats": PAIR_ENV_STATS,
        "ablation": ablation,
        "shape": list(stats_norm.shape),
        "max_pairs": max_pairs,
        "rows_with_pairs": int(np.sum(counts > 0)),
        "mean_pairs_per_row": float(np.mean(counts)),
        "max_pairs_observed_or_clipped": int(np.max(counts)) if len(counts) else 0,
        "rows_clipped": int(overflow),
    }
    return pair_ids, stats_norm.astype(np.float32), mean, std, summary


def load_shell_pair_env_features(
    frame: pd.DataFrame,
    shell_pair_env_cache: Path,
    train_mask: np.ndarray,
    *,
    max_pairs: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    cols = ["id", "cif_path", "pair_id", *SHELL_PAIR_ENV_STATS]
    env = pd.read_csv(shell_pair_env_cache, keep_default_na=False, usecols=lambda col: col in cols)
    env["id"] = env["id"].astype(str)
    env["cif_path"] = env["cif_path"].astype(str)
    env["pair_id"] = pd.to_numeric(env["pair_id"], errors="coerce").fillna(0).astype(np.int64)
    for stat in SHELL_PAIR_ENV_STATS:
        env[stat] = pd.to_numeric(env[stat], errors="coerce").fillna(0.0).astype(np.float32)
    sort_cols = ["id", "cif_path", "pair_frac", "shell1_bond_count_frac", "shell2_bond_count_frac", "shell3_bond_count_frac"]
    env = env.sort_values(sort_cols, ascending=[True, True, False, False, False, False])

    row_lookup = {
        (str(row.id), str(row.cif_path)): idx
        for idx, row in enumerate(frame[["id", "cif_path"]].itertuples(index=False))
    }
    max_pairs = int(max(1, max_pairs))
    pair_ids = np.zeros((len(frame), max_pairs), dtype=np.int64)
    pair_stats = np.zeros((len(frame), max_pairs, len(SHELL_PAIR_ENV_STATS)), dtype=np.float32)
    counts = np.zeros(len(frame), dtype=np.int32)
    overflow = 0
    for key, group in env.groupby(["id", "cif_path"], sort=False):
        row_idx = row_lookup.get((str(key[0]), str(key[1])))
        if row_idx is None:
            continue
        limited = group.head(max_pairs)
        n_pairs = len(limited)
        counts[row_idx] = n_pairs
        if len(group) > max_pairs:
            overflow += 1
        if n_pairs:
            pair_ids[row_idx, :n_pairs] = limited["pair_id"].to_numpy(dtype=np.int64)
            pair_stats[row_idx, :n_pairs, :] = limited[SHELL_PAIR_ENV_STATS].to_numpy(dtype=np.float32)

    mean = pair_stats[train_mask].reshape(-1, len(SHELL_PAIR_ENV_STATS)).mean(axis=0).astype(np.float32)
    std = pair_stats[train_mask].reshape(-1, len(SHELL_PAIR_ENV_STATS)).std(axis=0).astype(np.float32)
    passthrough = {"pair_frac", "same_pair", "shell1_bond_count_frac", "shell2_bond_count_frac", "shell3_bond_count_frac"}
    for idx, stat in enumerate(SHELL_PAIR_ENV_STATS):
        if stat in passthrough:
            mean[idx] = 0.0
            std[idx] = 1.0
    std[std < 1e-8] = 1.0
    stats_norm = pair_stats.copy()
    for idx, stat in enumerate(SHELL_PAIR_ENV_STATS):
        if stat not in passthrough:
            stats_norm[:, :, idx] = (pair_stats[:, :, idx] - mean[idx]) / std[idx]
    summary = {
        "cache": str(shell_pair_env_cache),
        "stats": SHELL_PAIR_ENV_STATS,
        "shape": list(stats_norm.shape),
        "max_pairs": max_pairs,
        "rows_with_pairs": int(np.sum(counts > 0)),
        "mean_pairs_per_row": float(np.mean(counts)),
        "max_pairs_observed_or_clipped": int(np.max(counts)) if len(counts) else 0,
        "rows_clipped": int(overflow),
    }
    return pair_ids, stats_norm.astype(np.float32), mean, std, summary


def load_chem_pair_env_features(
    frame: pd.DataFrame,
    chem_pair_env_cache: Path,
    train_mask: np.ndarray,
    *,
    max_pairs: int,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    mode = str(mode).lower()
    valid_modes = {"covnorm", "chemprop", "full", "noraw"}
    if mode not in valid_modes:
        raise ValueError(f"Unknown chem_pair_env_mode={mode!r}; expected one of {sorted(valid_modes)}")

    cols = ["id", "cif_path", "pair_id", *CHEM_PAIR_ENV_STATS]
    env = pd.read_csv(chem_pair_env_cache, keep_default_na=False, low_memory=False, usecols=lambda col: col in cols)
    env["id"] = env["id"].astype(str)
    env["cif_path"] = env["cif_path"].astype(str)
    env["pair_id"] = pd.to_numeric(env["pair_id"], errors="coerce").fillna(0).astype(np.int64)
    for stat in CHEM_PAIR_ENV_STATS:
        env[stat] = pd.to_numeric(env[stat], errors="coerce").fillna(0.0).astype(np.float32)
    sort_cols = ["id", "cif_path", "pair_frac", "bond_count_frac"]
    env = env.sort_values(sort_cols, ascending=[True, True, False, False])

    row_lookup = {
        (str(row.id), str(row.cif_path)): idx
        for idx, row in enumerate(frame[["id", "cif_path"]].itertuples(index=False))
    }
    max_pairs = int(max(1, max_pairs))
    pair_ids = np.zeros((len(frame), max_pairs), dtype=np.int64)
    pair_stats = np.zeros((len(frame), max_pairs, len(CHEM_PAIR_ENV_STATS)), dtype=np.float32)
    counts = np.zeros(len(frame), dtype=np.int32)
    overflow = 0
    for key, group in env.groupby(["id", "cif_path"], sort=False):
        row_idx = row_lookup.get((str(key[0]), str(key[1])))
        if row_idx is None:
            continue
        limited = group.head(max_pairs)
        n_pairs = len(limited)
        counts[row_idx] = n_pairs
        if len(group) > max_pairs:
            overflow += 1
        if n_pairs:
            pair_ids[row_idx, :n_pairs] = limited["pair_id"].to_numpy(dtype=np.int64)
            pair_stats[row_idx, :n_pairs, :] = limited[CHEM_PAIR_ENV_STATS].to_numpy(dtype=np.float32)

    passthrough = {
        "pair_frac",
        "bond_count_frac",
        "same_pair",
        "same_period",
        "same_group",
        "metal_pair_type",
    }
    mean = pair_stats[train_mask].reshape(-1, len(CHEM_PAIR_ENV_STATS)).mean(axis=0).astype(np.float32)
    std = pair_stats[train_mask].reshape(-1, len(CHEM_PAIR_ENV_STATS)).std(axis=0).astype(np.float32)
    for idx, stat in enumerate(CHEM_PAIR_ENV_STATS):
        if stat in passthrough:
            mean[idx] = 0.0
            std[idx] = 1.0
    std[std < 1e-8] = 1.0
    stats_norm = pair_stats.copy()
    for idx, stat in enumerate(CHEM_PAIR_ENV_STATS):
        if stat not in passthrough:
            stats_norm[:, :, idx] = (pair_stats[:, :, idx] - mean[idx]) / std[idx]

    cov_norm = {"bond_mean_cov_norm", "bond_std_cov_norm", "bond_min_cov_norm", "bond_max_cov_norm"}
    atomic_norm = {"bond_mean_atomic_norm", "bond_std_atomic_norm", "bond_min_atomic_norm", "bond_max_atomic_norm"}
    chem_props = {
        "electronegativity_diff",
        "atomic_radius_ratio",
        "covalent_radius_sum",
        "atomic_radius_sum",
        "period_diff",
        "group_diff",
        "same_period",
        "same_group",
        "metal_pair_type",
    }
    keep: set[str]
    if mode == "covnorm":
        keep = {"pair_frac", "bond_count_frac", "same_pair", *cov_norm}
    elif mode == "chemprop":
        keep = {"pair_frac", "bond_count_frac", "same_pair", *chem_props}
    elif mode in {"full", "noraw"}:
        keep = {"pair_frac", "bond_count_frac", "same_pair", *cov_norm, *atomic_norm, *chem_props}
    for idx, stat in enumerate(CHEM_PAIR_ENV_STATS):
        if stat not in keep:
            stats_norm[:, :, idx] = 0.0

    summary = {
        "cache": str(chem_pair_env_cache),
        "stats": CHEM_PAIR_ENV_STATS,
        "mode": mode,
        "shape": list(stats_norm.shape),
        "max_pairs": max_pairs,
        "rows_with_pairs": int(np.sum(counts > 0)),
        "mean_pairs_per_row": float(np.mean(counts)),
        "max_pairs_observed_or_clipped": int(np.max(counts)) if len(counts) else 0,
        "rows_clipped": int(overflow),
        "active_stats": [stat for stat in CHEM_PAIR_ENV_STATS if stat in keep],
        "disabled_stats": [stat for stat in CHEM_PAIR_ENV_STATS if stat not in keep],
    }
    return pair_ids, stats_norm.astype(np.float32), mean, std, summary


def load_dual_foundation_features(
    frame: pd.DataFrame,
    dual_calibrated_predictions: Path,
    train_mask: np.ndarray,
    *,
    ablation: str = "none",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    cols = [
        "split",
        "id",
        "model_energy_per_atom",
        "pred_element_reference",
        "pred_affine_energy_composition",
        "primary_missing",
    ]
    aux = pd.read_csv(dual_calibrated_predictions, usecols=lambda col: col in cols)
    needed = {"split", "id", "model_energy_per_atom", "pred_element_reference", "pred_affine_energy_composition"}
    if not needed.issubset(aux.columns):
        raise KeyError(f"{dual_calibrated_predictions} is missing columns {sorted(needed - set(aux.columns))}")
    aux = aux.rename(
        columns={
            "model_energy_per_atom": "aux_model_energy_per_atom",
            "pred_element_reference": "aux_element_reference",
            "pred_affine_energy_composition": "aux_affine_reference",
            "primary_missing": "aux_primary_missing",
        }
    )
    aux["split"] = aux["split"].astype(str)
    aux["id"] = aux["id"].astype(str)
    aux_duplicate_rows = int(aux.duplicated(["split", "id"]).sum())
    if aux_duplicate_rows:
        aux = aux.drop_duplicates(["split", "id"], keep="last")
    base = frame[
        [
            "split",
            "id",
            "model_energy_per_atom",
            "pred_element_reference",
            "pred_affine_energy_composition",
            *([ "primary_missing" ] if "primary_missing" in frame.columns else []),
        ]
    ].copy()
    base["split"] = base["split"].astype(str)
    base["id"] = base["id"].astype(str)
    merged = base.merge(aux, on=["split", "id"], how="left")
    primary_missing = (
        pd.to_numeric(merged["primary_missing"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        if "primary_missing" in merged.columns
        else np.zeros(len(merged), dtype=np.float32)
    )
    if "aux_primary_missing" in merged.columns:
        aux_missing = pd.to_numeric(merged["aux_primary_missing"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    else:
        aux_missing = merged["aux_model_energy_per_atom"].isna().to_numpy(dtype=np.float32)
    for col in ["model_energy_per_atom", "pred_element_reference", "pred_affine_energy_composition"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
        fill = float(np.nanmean(merged.loc[train_mask, col].to_numpy(dtype=np.float32)))
        if not np.isfinite(fill):
            fill = 0.0
        merged[col] = merged[col].fillna(fill)
    for col in ["aux_model_energy_per_atom", "aux_element_reference", "aux_affine_reference"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
        fill = float(np.nanmean(merged.loc[train_mask, col].to_numpy(dtype=np.float32)))
        if not np.isfinite(fill):
            fill = 0.0
        merged[col] = merged[col].fillna(fill)

    primary_energy = merged["model_energy_per_atom"].to_numpy(dtype=np.float32)
    primary_element_ref = merged["pred_element_reference"].to_numpy(dtype=np.float32)
    primary_affine_ref = merged["pred_affine_energy_composition"].to_numpy(dtype=np.float32)
    aux_energy = merged["aux_model_energy_per_atom"].to_numpy(dtype=np.float32)
    aux_element_ref = merged["aux_element_reference"].to_numpy(dtype=np.float32)
    aux_affine_ref = merged["aux_affine_reference"].to_numpy(dtype=np.float32)
    raw = np.column_stack(
        [
            aux_energy,
            aux_element_ref,
            aux_affine_ref,
            primary_energy - aux_energy,
            np.abs(primary_energy - aux_energy),
            primary_element_ref - aux_element_ref,
            np.abs(primary_element_ref - aux_element_ref),
            primary_affine_ref - aux_affine_ref,
            np.abs(primary_affine_ref - aux_affine_ref),
            aux_missing,
            primary_missing,
        ]
    ).astype(np.float32)
    mean = raw[train_mask].mean(axis=0).astype(np.float32)
    std = raw[train_mask].std(axis=0).astype(np.float32)
    for idx, stat in enumerate(DUAL_FOUNDATION_STATS):
        if stat in DUAL_FOUNDATION_PASSTHROUGH_STATS:
            mean[idx] = 0.0
    std[std < 1e-8] = 1.0
    for idx, stat in enumerate(DUAL_FOUNDATION_STATS):
        if stat in DUAL_FOUNDATION_PASSTHROUGH_STATS:
            std[idx] = 1.0
    stats_norm = raw.copy()
    for idx, stat in enumerate(DUAL_FOUNDATION_STATS):
        if stat not in DUAL_FOUNDATION_PASSTHROUGH_STATS:
            stats_norm[:, idx] = (raw[:, idx] - mean[idx]) / std[idx]
    if np.any(primary_missing > 0.5):
        for stat in DUAL_FOUNDATION_PRIMARY_DEPENDENT_STATS:
            idx = DUAL_FOUNDATION_STATS.index(stat)
            stats_norm[primary_missing > 0.5, idx] = 0.0
    mode = str(ablation).lower()
    if mode not in DUAL_FOUNDATION_ACTIVE_STATS_BY_MODE:
        raise ValueError(
            f"Unknown dual foundation ablation {ablation!r}; "
            f"expected one of {sorted(DUAL_FOUNDATION_ACTIVE_STATS_BY_MODE)}"
        )
    active = DUAL_FOUNDATION_ACTIVE_STATS_BY_MODE[mode]
    disabled = [stat for stat in DUAL_FOUNDATION_STATS if stat not in active]
    if disabled:
        for idx, stat in enumerate(DUAL_FOUNDATION_STATS):
            if stat not in active:
                stats_norm[:, idx] = 0.0
    summary = {
        "enabled": True,
        "calibrated_predictions": str(dual_calibrated_predictions),
        "stats": DUAL_FOUNDATION_STATS,
        "ablation": mode,
        "active_stats": [stat for stat in DUAL_FOUNDATION_STATS if stat in active],
        "disabled_stats": disabled,
        "shape": list(stats_norm.shape),
        "aux_duplicate_rows_dropped": aux_duplicate_rows,
        "missing_rows": int(aux_missing.sum()),
        "primary_missing_rows": int(primary_missing.sum()),
        "mean_abs_energy_diff_train_mev_atom": float(np.mean(np.abs(raw[train_mask, 3])) * 1000.0),
        "mean_abs_element_ref_diff_train_mev_atom": float(np.mean(np.abs(raw[train_mask, 5])) * 1000.0),
    }
    return stats_norm.astype(np.float32), mean, std, summary


def load_tail_feature_cache(
    frame: pd.DataFrame,
    tail_feature_cache: Path,
    train_mask: np.ndarray,
    *,
    columns: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], dict[str, object]]:
    requested = [col.strip() for col in str(columns).split(",") if col.strip()]
    if not requested:
        raise ValueError("use_tail_features=True requires non-empty --tail-feature-cols")
    cols = ["id", "cif_path", *requested]
    cache = pd.read_csv(tail_feature_cache, keep_default_na=False, low_memory=False, usecols=lambda col: col in cols)
    for key in ["id", "cif_path"]:
        if key not in cache.columns:
            raise KeyError(f"{tail_feature_cache} is missing key column {key!r}")
        cache[key] = cache[key].astype(str)
    missing = [col for col in requested if col not in cache.columns]
    if missing:
        raise KeyError(f"{tail_feature_cache} is missing tail feature columns: {missing}")
    cache = cache.drop_duplicates(["id", "cif_path"], keep="last")
    keys = frame[["id", "cif_path"]].copy()
    keys["id"] = keys["id"].astype(str)
    keys["cif_path"] = keys["cif_path"].astype(str)
    merged = keys.merge(cache, on=["id", "cif_path"], how="left")
    values = merged[requested].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    missing_rows = np.isnan(values).any(axis=1)
    fill = np.nanmean(values[train_mask], axis=0).astype(np.float32)
    fill = np.where(np.isfinite(fill), fill, 0.0).astype(np.float32)
    inds = np.where(~np.isfinite(values))
    if len(inds[0]):
        values[inds] = np.take(fill, inds[1])
    mean = values[train_mask].mean(axis=0).astype(np.float32)
    std = values[train_mask].std(axis=0).astype(np.float32)
    std[std < 1e-8] = 1.0
    norm = ((values - mean) / std).astype(np.float32)
    summary = {
        "enabled": True,
        "cache": str(tail_feature_cache),
        "columns": requested,
        "shape": list(norm.shape),
        "missing_rows": int(missing_rows.sum()),
    }
    return norm, mean, std, requested, summary


def fegx1_prepare_structure(
    frame: pd.DataFrame,
    train_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    scalars = frame[STRUCTURE_SCALAR_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
    fill_values = np.nanmean(scalars[train_mask], axis=0)
    fill_values = np.where(np.isfinite(fill_values), fill_values, 0.0).astype(np.float32)
    inds = np.where(~np.isfinite(scalars))
    if len(inds[0]):
        scalars[inds] = np.take(fill_values, inds[1])
    mean = scalars[train_mask].mean(axis=0).astype(np.float32)
    std = scalars[train_mask].std(axis=0).astype(np.float32)
    std[std < 1e-8] = 1.0

    sg = pd.to_numeric(frame["spacegroup_number"], errors="coerce").fillna(0).astype(int).clip(0, 230).to_numpy(dtype=np.int64)
    # Shift physical 1..230 to 1..230 and keep 0 for unknown; embedding has 232 rows.
    crystal_map = {name: idx for idx, name in enumerate(CRYSTAL_SYSTEMS)}
    crystal = (
        frame["crystal_system"]
        .astype(str)
        .str.lower()
        .map(crystal_map)
        .fillna(0)
        .astype(int)
        .to_numpy(dtype=np.int64)
    )
    return scalars, sg, crystal, mean, std, crystal_map


def fegx1_metrics(frame: pd.DataFrame, pred_col: str) -> dict[str, object]:
    out: dict[str, object] = {}
    for split, group in frame.groupby("split"):
        out[split] = {
            "overall": regression_metrics(group["y_true"].to_numpy(), group[pred_col].to_numpy()),
            "slices": slice_metrics(group, pred_col, "y_true"),
        }
    return out


def grouped_regression_metrics(
    frame: pd.DataFrame,
    *,
    pred_col: str,
    target_col: str,
    group_col: str,
) -> dict[str, dict[str, float | int]]:
    if group_col not in frame.columns:
        return {}
    out: dict[str, dict[str, float | int]] = {}
    for group_name, group in frame.groupby(group_col, sort=True, dropna=False):
        label = str(group_name).strip()
        if not label or label.lower() == "nan":
            continue
        out[label] = regression_metrics(
            group[target_col].to_numpy(dtype=np.float64),
            group[pred_col].to_numpy(dtype=np.float64),
        )
    return out


def _metric_slug(text: str) -> str:
    slug = "".join(ch if ch.isalnum() else "_" for ch in str(text).strip().lower())
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "unknown"


def _hybrid_coverage_rows(primary_rows: int, primary_ratio: float) -> int:
    if primary_rows <= 0 or primary_ratio >= 1.0:
        return 0
    return max(1, int(round(float(primary_rows) * (1.0 - primary_ratio) / max(primary_ratio, 1e-6))))


def build_train_batches(
    frame: pd.DataFrame,
    *,
    train_indices: np.ndarray,
    batch_size: int,
    seed: int,
    epoch: int,
    sampling_mode: str,
    hybrid_primary_ratio: float,
    hybrid_group_col: str,
) -> tuple[list[np.ndarray], dict[str, object]]:
    rng = np.random.default_rng(int(seed) + int(epoch))
    shuffled = np.asarray(train_indices, dtype=np.int64).copy()
    rng.shuffle(shuffled)
    mode = str(sampling_mode or "shuffle").strip().lower()

    if mode in {"", "shuffle"}:
        batches = [shuffled[start : start + int(batch_size)] for start in range(0, len(shuffled), int(batch_size))]
        summary = {
            "enabled": False,
            "mode": "shuffle",
            "epoch_batches": int(len(batches)),
            "epoch_primary_rows": int(len(shuffled)),
            "epoch_coverage_rows": 0,
            "epoch_total_rows": int(sum(len(batch) for batch in batches)),
            "epoch_coverage_fraction": 0.0,
        }
        return batches, summary

    if mode != "hybrid_coverage":
        raise ValueError(f"Unsupported train_sampling_mode={sampling_mode!r}")
    if not 0.0 < float(hybrid_primary_ratio) <= 1.0:
        raise ValueError("hybrid_sampling_primary_ratio must be in (0, 1].")
    if float(hybrid_primary_ratio) >= 1.0:
        return build_train_batches(
            frame,
            train_indices=train_indices,
            batch_size=batch_size,
            seed=seed,
            epoch=epoch,
            sampling_mode="shuffle",
            hybrid_primary_ratio=1.0,
            hybrid_group_col=hybrid_group_col,
        )
    if hybrid_group_col not in frame.columns:
        raise KeyError(
            f"Hybrid coverage sampling requires column {hybrid_group_col!r} in calibrated predictions frame."
        )

    primary_per_full_batch = max(1, min(int(batch_size), int(round(int(batch_size) * float(hybrid_primary_ratio)))))
    group_frame = frame.loc[shuffled, [hybrid_group_col]]
    missing_group_count = int(group_frame[hybrid_group_col].astype(str).str.strip().eq("").sum())
    if missing_group_count:
        raise ValueError(
            f"Hybrid coverage sampling found {missing_group_count} train rows with empty "
            f"{hybrid_group_col!r}; fix dataset/calibration metadata before training."
        )
    group_to_indices: dict[str, np.ndarray] = {}
    for group_name, group in group_frame.groupby(hybrid_group_col, sort=True, dropna=False):
        label = str(group_name).strip()
        if not label or label.lower() == "nan":
            continue
        indices = group.index.to_numpy(dtype=np.int64)
        if len(indices):
            group_to_indices[label] = indices
    if not group_to_indices:
        raise ValueError(
            f"Hybrid coverage sampling could not build any non-empty groups from column {hybrid_group_col!r}."
        )

    group_order = list(group_to_indices.keys())
    rng.shuffle(group_order)
    group_cursor = 0
    group_pools: dict[str, np.ndarray] = {}
    group_pos: dict[str, int] = {}
    group_restarts: Counter[str] = Counter()
    group_usage: Counter[str] = Counter()
    for label, indices in group_to_indices.items():
        pool = indices.copy()
        rng.shuffle(pool)
        group_pools[label] = pool
        group_pos[label] = 0

    fallback_pool = shuffled.copy()
    rng.shuffle(fallback_pool)
    fallback_pos = 0

    def next_group_name() -> str:
        nonlocal group_cursor
        label = group_order[group_cursor % len(group_order)]
        group_cursor += 1
        return label

    def next_from_group(label: str) -> int:
        pool = group_pools[label]
        pos = group_pos[label]
        if pos >= len(pool):
            pool = group_to_indices[label].copy()
            rng.shuffle(pool)
            group_pools[label] = pool
            group_pos[label] = 0
            group_restarts[label] += 1
        index = int(group_pools[label][group_pos[label]])
        group_pos[label] += 1
        return index

    def next_fallback() -> int:
        nonlocal fallback_pool, fallback_pos
        if fallback_pos >= len(fallback_pool):
            fallback_pool = shuffled.copy()
            rng.shuffle(fallback_pool)
            fallback_pos = 0
        index = int(fallback_pool[fallback_pos])
        fallback_pos += 1
        return index

    batches: list[np.ndarray] = []
    coverage_rows_total = 0
    duplicates_avoided = 0
    for start in range(0, len(shuffled), primary_per_full_batch):
        primary = shuffled[start : start + primary_per_full_batch]
        coverage_needed = _hybrid_coverage_rows(len(primary), float(hybrid_primary_ratio))
        seen = set(int(idx) for idx in primary.tolist())
        coverage: list[int] = []
        guard = 0
        max_guard = max(100, coverage_needed * 50)
        while len(coverage) < coverage_needed and guard < max_guard:
            label = next_group_name()
            candidate = next_from_group(label)
            guard += 1
            if candidate in seen:
                duplicates_avoided += 1
                continue
            coverage.append(candidate)
            seen.add(candidate)
            group_usage[label] += 1
        while len(coverage) < coverage_needed and guard < max_guard * 2:
            candidate = next_fallback()
            guard += 1
            if candidate in seen:
                duplicates_avoided += 1
                continue
            coverage.append(candidate)
            seen.add(candidate)
            group_usage["__fallback__"] += 1
        coverage_rows_total += len(coverage)
        if coverage:
            batch = np.concatenate([primary, np.asarray(coverage, dtype=np.int64)])
        else:
            batch = primary
        batches.append(batch.astype(np.int64, copy=False))

    summary = {
        "enabled": True,
        "mode": "hybrid_coverage",
        "group_col": hybrid_group_col,
        "primary_ratio": float(hybrid_primary_ratio),
        "batch_size": int(batch_size),
        "primary_rows_per_full_batch": int(primary_per_full_batch),
        "coverage_rows_per_full_batch": int(
            _hybrid_coverage_rows(primary_per_full_batch, float(hybrid_primary_ratio))
        ),
        "n_groups": int(len(group_to_indices)),
        "group_counts": {label: int(len(indices)) for label, indices in group_to_indices.items()},
        "epoch_batches": int(len(batches)),
        "epoch_primary_rows": int(len(shuffled)),
        "epoch_coverage_rows": int(coverage_rows_total),
        "epoch_total_rows": int(sum(len(batch) for batch in batches)),
        "epoch_coverage_fraction": float(coverage_rows_total / max(1, sum(len(batch) for batch in batches))),
        "epoch_group_usage": {label: int(count) for label, count in group_usage.items()},
        "epoch_group_pool_restarts": {label: int(count) for label, count in group_restarts.items() if count > 0},
        "duplicates_avoided_in_epoch": int(duplicates_avoided),
    }
    return batches, summary


def build_formula_rank_pairs(
    frame: pd.DataFrame,
    *,
    y_true: np.ndarray,
    train_mask: np.ndarray,
    margin_mev: float,
    pair_mode: str,
    max_pairs_per_formula: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    if "formula" not in frame.columns:
        return (
            np.asarray([], dtype=np.int64),
            np.asarray([], dtype=np.int64),
            np.asarray([], dtype=np.float32),
            {"enabled": False, "reason": "missing_formula_column", "n_pairs": 0},
        )

    margin_ev = float(margin_mev) / 1000.0
    rng = np.random.default_rng(seed)
    left: list[int] = []
    right: list[int] = []
    signs: list[float] = []
    group_count = 0
    candidate_count = 0
    kept_per_formula: list[int] = []
    pair_mode = str(pair_mode).lower()

    train_frame = frame.loc[train_mask, ["formula"]]
    for _, group in train_frame.groupby("formula", sort=False):
        indices = group.index.to_numpy(dtype=np.int64)
        if len(indices) < 2:
            continue
        y = y_true[indices]
        group_pairs: list[tuple[int, int, float]] = []
        if pair_mode == "top1":
            local_best = int(np.argmin(y))
            for local_j in range(len(indices)):
                if local_j == local_best or abs(float(y[local_j] - y[local_best])) <= margin_ev:
                    continue
                group_pairs.append((int(indices[local_best]), int(indices[local_j]), 1.0))
        else:
            for local_i in range(len(indices) - 1):
                deltas = y[local_i + 1 :] - y[local_i]
                keep = np.flatnonzero(np.abs(deltas) > margin_ev)
                for offset in keep:
                    local_j = local_i + 1 + int(offset)
                    sign = 1.0 if y[local_j] > y[local_i] else -1.0
                    group_pairs.append((int(indices[local_i]), int(indices[local_j]), sign))
        if not group_pairs:
            continue
        group_count += 1
        candidate_count += len(group_pairs)
        if len(group_pairs) > max_pairs_per_formula:
            picked = rng.choice(len(group_pairs), size=max_pairs_per_formula, replace=False)
            group_pairs = [group_pairs[int(idx)] for idx in picked]
        kept_per_formula.append(len(group_pairs))
        for i, j, sign in group_pairs:
            left.append(i)
            right.append(j)
            signs.append(sign)

    summary = {
        "enabled": True,
        "n_pairs": int(len(left)),
        "candidate_pairs": int(candidate_count),
        "n_formula_groups": int(group_count),
        "margin_mev": float(margin_mev),
        "pair_mode": pair_mode,
        "max_pairs_per_formula": int(max_pairs_per_formula),
        "mean_pairs_per_formula": float(np.mean(kept_per_formula)) if kept_per_formula else 0.0,
        "max_pairs_in_formula": int(np.max(kept_per_formula)) if kept_per_formula else 0,
    }
    return (
        np.asarray(left, dtype=np.int64),
        np.asarray(right, dtype=np.int64),
        np.asarray(signs, dtype=np.float32),
        summary,
    )


def build_tail_sample_weights(
    *,
    y_true: np.ndarray,
    nelements: np.ndarray,
    element_reference: np.ndarray,
    train_mask: np.ndarray,
    config: FEGX1Config,
) -> tuple[np.ndarray, dict[str, object]]:
    weights = np.ones(len(y_true), dtype=np.float32)
    mode = str(config.tail_weight_mode).lower()
    if mode == "none" or config.tail_max_weight <= 1.0:
        return weights, {
            "mode": mode,
            "enabled": False,
            "min_weight": 1.0,
            "mean_train_weight": 1.0,
            "max_train_weight": 1.0,
        }

    ref_abs = np.abs(y_true - element_reference)
    train_ref_abs = ref_abs[train_mask]
    finite_ref = train_ref_abs[np.isfinite(train_ref_abs)]
    ref_threshold = float(np.quantile(finite_ref, config.tail_ref_quantile)) if len(finite_ref) else float("inf")

    if mode in {"positive_binary", "ref_positive_binary"}:
        weights[train_mask] += config.tail_positive_increment * (y_true[train_mask] >= 0.0).astype(np.float32)
        weights[train_mask] += config.tail_binary_increment * (nelements[train_mask] == 2).astype(np.float32)
    if mode in {"ref", "ref_positive_binary"}:
        weights[train_mask] += config.tail_ref_increment * (ref_abs[train_mask] >= ref_threshold).astype(np.float32)

    weights = np.minimum(weights, float(config.tail_max_weight)).astype(np.float32)
    train_weights = weights[train_mask]
    return weights, {
        "mode": mode,
        "enabled": True,
        "ref_quantile": float(config.tail_ref_quantile),
        "ref_threshold_mev_atom": float(ref_threshold * 1000.0),
        "tail_max_weight": float(config.tail_max_weight),
        "positive_increment": float(config.tail_positive_increment),
        "binary_increment": float(config.tail_binary_increment),
        "ref_increment": float(config.tail_ref_increment),
        "min_train_weight": float(np.min(train_weights)) if len(train_weights) else float("nan"),
        "mean_train_weight": float(np.mean(train_weights)) if len(train_weights) else float("nan"),
        "max_train_weight": float(np.max(train_weights)) if len(train_weights) else float("nan"),
        "weighted_train_rows": int(np.sum(train_weights > 1.0)) if len(train_weights) else 0,
    }


def build_router_anchor_mask(
    *,
    dual_foundation_stats: np.ndarray,
    train_mask: np.ndarray,
    config: FEGX1Config,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    mask = np.zeros(len(train_mask), dtype=bool)
    risk_score = np.zeros(len(train_mask), dtype=np.float32)
    summary: dict[str, object] = {
        "enabled": False,
        "weight": float(config.router_anchor_weight),
        "risk_branch_anchor_weight": float(config.risk_branch_anchor_weight),
        "risk_quantile": float(config.router_anchor_risk_quantile),
        "target_expert": int(config.router_anchor_target_expert),
    }
    needs_router_anchor = config.router_anchor_weight > 0.0
    needs_risk_branch_anchor = bool(config.use_risk_residual_branch and config.risk_branch_anchor_weight > 0.0)
    if not needs_router_anchor and not needs_risk_branch_anchor:
        summary["reason"] = "no_anchor_loss_enabled"
        return mask, risk_score, summary
    if needs_router_anchor and config.residual_head_type != "moe":
        summary["reason"] = "requires_moe"
        return mask, risk_score, summary
    if not config.use_dual_foundation:
        summary["reason"] = "requires_dual_foundation"
        return mask, risk_score, summary
    if needs_router_anchor and (
        config.router_anchor_target_expert < 0 or config.router_anchor_target_expert >= config.num_experts
    ):
        raise ValueError(
            "router_anchor_target_expert must be in [0, num_experts); "
            f"got {config.router_anchor_target_expert} for {config.num_experts} experts"
        )
    if not 0.0 < config.router_anchor_risk_quantile < 1.0:
        raise ValueError("router_anchor_risk_quantile must be between 0 and 1")

    risk_cols = [
        DUAL_FOUNDATION_STATS.index("energy_abs_diff"),
        DUAL_FOUNDATION_STATS.index("element_reference_abs_diff"),
        DUAL_FOUNDATION_STATS.index("affine_reference_abs_diff"),
    ]
    risk_score = np.sum(dual_foundation_stats[:, risk_cols], axis=1).astype(np.float32)
    train_scores = risk_score[train_mask]
    finite_scores = train_scores[np.isfinite(train_scores)]
    if len(finite_scores) == 0 or float(np.std(finite_scores)) < 1e-8:
        summary["reason"] = "constant_or_missing_risk_score"
        return mask, risk_score, summary

    threshold = float(np.quantile(finite_scores, config.router_anchor_risk_quantile))
    mask = train_mask & np.isfinite(risk_score) & (risk_score >= threshold)
    summary.update(
        {
            "enabled": True,
            "risk_stats": [
                DUAL_FOUNDATION_STATS[idx] for idx in risk_cols
            ],
            "risk_threshold": threshold,
            "n_train_anchor": int(np.sum(mask & train_mask)),
            "train_anchor_fraction": float(np.mean(mask[train_mask])) if int(np.sum(train_mask)) else 0.0,
            "mean_train_risk_score": float(np.mean(finite_scores)),
            "mean_anchor_risk_score": float(np.mean(risk_score[mask])) if int(np.sum(mask)) else float("nan"),
        }
    )
    return mask, risk_score, summary


def apply_primary_missing_to_dual_stats_tensor(
    dual_foundation_stats: torch.Tensor | None,
    primary_missing: torch.Tensor | None,
) -> torch.Tensor | None:
    if dual_foundation_stats is None or dual_foundation_stats.numel() == 0 or primary_missing is None:
        return dual_foundation_stats
    primary_missing = primary_missing.reshape(-1).to(dtype=dual_foundation_stats.dtype, device=dual_foundation_stats.device)
    if primary_missing.numel() == 0:
        return dual_foundation_stats
    masked = dual_foundation_stats.clone()
    primary_missing_idx = DUAL_FOUNDATION_STATS.index("primary_missing")
    masked[:, primary_missing_idx] = primary_missing
    if bool(torch.any(primary_missing > 0.5).detach().cpu()):
        for stat in DUAL_FOUNDATION_PRIMARY_DEPENDENT_STATS:
            idx = DUAL_FOUNDATION_STATS.index(stat)
            masked[primary_missing > 0.5, idx] = 0.0
    return masked


SOURCE_SPECIFIC_INIT_KEYS = {
    "scalar_mean",
    "scalar_std",
    "residual_mean",
    "residual_std",
    "dual_foundation_mean",
    "dual_foundation_std",
}


def initialize_fegx1_from_checkpoint(
    model: FEGX1Model,
    checkpoint_path: str,
    *,
    noise_std: float,
    transfer_mode: str = "all",
) -> dict[str, object]:
    if not checkpoint_path:
        return {"enabled": False}
    if transfer_mode not in {"all", "body_only"}:
        raise ValueError(f"Unsupported init transfer mode: {transfer_mode}")
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"init checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    current = model.state_dict()
    direct: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    expert_loaded = 0
    expanded_tensors = 0
    checkpoint_has_moe = any(key.startswith("residual_head.experts.") for key in state)

    for key, value in state.items():
        if transfer_mode == "body_only" and (
            key.startswith("element_reference.")
            or key.startswith("affine_reference.")
            or key in SOURCE_SPECIFIC_INIT_KEYS
        ):
            skipped.append(key)
            continue
        if key.startswith("residual_head.") and not checkpoint_has_moe and isinstance(model.residual_head, ResidualMoEHead):
            continue
        if key in current and current[key].shape == value.shape:
            direct[key] = value
        elif key in current and value.ndim == 2 and current[key].ndim == 2 and current[key].shape[0] == value.shape[0] and current[key].shape[1] > value.shape[1]:
            expanded = current[key].clone()
            expanded[:, : value.shape[1]] = value
            expanded[:, value.shape[1] :] = 0.0
            direct[key] = expanded
            expanded_tensors += 1
        else:
            skipped.append(key)
    model.load_state_dict(direct, strict=False)

    if isinstance(model.residual_head, ResidualMoEHead) and checkpoint_has_moe:
        expert_loaded = int(model.residual_head.num_experts)
    elif isinstance(model.residual_head, ResidualMoEHead):
        expert_state = {
            key.removeprefix("residual_head."): value
            for key, value in state.items()
            if key.startswith("residual_head.")
        }
        if expert_state:
            for idx, expert in enumerate(model.residual_head.experts):
                copied = {key: value.clone() for key, value in expert_state.items()}
                if noise_std > 0.0 and idx > 0:
                    generator = torch.Generator().manual_seed(20260528 + idx)
                    for key, value in copied.items():
                        if value.is_floating_point():
                            copied[key] = value + torch.randn(value.shape, generator=generator, dtype=value.dtype) * float(noise_std)
                expert.load_state_dict(copied, strict=True)
                expert_loaded += 1

    return {
        "enabled": True,
        "checkpoint": str(path),
        "loaded_non_residual_tensors": int(len(direct)),
        "checkpoint_has_moe": bool(checkpoint_has_moe),
        "skipped_tensors": int(len(skipped)),
        "expanded_input_tensors": int(expanded_tensors),
        "experts_initialized_from_residual_head": int(expert_loaded),
        "noise_std": float(noise_std),
        "transfer_mode": transfer_mode,
    }


def moe_regularization(router_weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if router_weights.numel() == 0:
        zero = torch.zeros((), device=router_weights.device)
        return zero, zero, zero
    mean_usage = router_weights.mean(dim=0)
    target = torch.full_like(mean_usage, 1.0 / router_weights.shape[1])
    load_balance = torch.sum((mean_usage - target).square())
    entropy = -torch.sum(router_weights * torch.log(router_weights.clamp_min(1e-8)), dim=1).mean()
    max_usage = torch.max(mean_usage)
    return load_balance, entropy, max_usage


def run_fegx0_training(
    *,
    calibrated_predictions: Path,
    feature_cache: Path,
    coefficient_csv: Path,
    out_dir: Path,
    config: FEGX0Config,
) -> dict[str, object]:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = load_fegx0_frame(calibrated_predictions, feature_cache)
    frame["split"] = frame["split"].astype(str)
    train_mask = frame["split"].to_numpy() == "train"
    val_mask = frame["split"].to_numpy() == "val"

    coeffs = load_reference_coefficients(coefficient_csv)
    x_comp = frame[X_COLS].to_numpy(dtype=np.float32)
    foundation = frame["model_energy_per_atom"].to_numpy(dtype=np.float32)
    natoms = frame["natoms"].to_numpy(dtype=np.float32)
    nelements = frame["nelements"].to_numpy(dtype=np.float32)
    y_true = frame["y_true"].to_numpy(dtype=np.float32)

    element_ref = foundation - x_comp @ coeffs["element_mu"] - float(coeffs["element_bias"])
    affine_ref = float(coeffs["affine_alpha"]) * foundation + x_comp @ coeffs["affine_beta"] + float(coeffs["affine_bias"])
    x_raw = fegx0_raw_features(
        composition=x_comp,
        foundation_energy=foundation,
        natoms=natoms,
        nelements=nelements,
        element_reference=element_ref,
        affine_reference=affine_ref,
    )
    feature_mean = x_raw[train_mask].mean(axis=0).astype(np.float32)
    feature_std = x_raw[train_mask].std(axis=0).astype(np.float32)
    feature_std[feature_std < 1e-8] = 1.0
    residual = y_true - element_ref
    residual_mean = float(residual[train_mask].mean())
    residual_std = float(residual[train_mask].std())
    if residual_std < 1e-8:
        residual_std = 1.0

    device = torch.device(config.device if torch.cuda.is_available() and config.device.startswith("cuda") else "cpu")
    trainable_reference = config.reference_lr > 0.0
    model = FEGX0Model(
        element_mu=coeffs["element_mu"],
        element_bias=float(coeffs["element_bias"]),
        affine_alpha=float(coeffs["affine_alpha"]),
        affine_beta=coeffs["affine_beta"],
        affine_bias=float(coeffs["affine_bias"]),
        feature_mean=feature_mean,
        feature_std=feature_std,
        residual_mean=residual_mean,
        residual_std=residual_std,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        trainable_reference=trainable_reference,
    ).to(device)

    head_params = list(model.residual_head.parameters())
    ref_params = [param for name, param in model.named_parameters() if not name.startswith("residual_head.")]
    param_groups: list[dict[str, object]] = [{"params": head_params, "lr": config.lr, "weight_decay": config.weight_decay}]
    if trainable_reference:
        param_groups.append({"params": ref_params, "lr": config.reference_lr, "weight_decay": 0.0})
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, config.epochs), eta_min=config.lr * 0.05)
    loss_fn = nn.SmoothL1Loss(beta=config.huber_beta)

    tensors = {
        "composition": torch.from_numpy(x_comp),
        "foundation": torch.from_numpy(foundation),
        "natoms": torch.from_numpy(natoms),
        "nelements": torch.from_numpy(nelements),
        "y_true": torch.from_numpy(y_true),
    }
    train_indices = np.flatnonzero(train_mask)
    val_indices = np.flatnonzero(val_mask)
    has_val = len(val_indices) > 0

    def predict_indices(indices: np.ndarray) -> dict[str, np.ndarray]:
        model.eval()
        chunks: dict[str, list[np.ndarray]] = {
            "pred": [],
            "base_pred": [],
            "adapter_delta": [],
            "anchor_component": [],
            "anchor_alpha": [],
            "grouped_structure_delta": [],
            "grouped_structure_raw_delta": [],
            "grouped_structure_magnitude": [],
            "grouped_structure_weights": [],
            "element_reference": [],
            "dual_reference": [],
            "aux_element_reference": [],
            "dual_reference_gate": [],
            "affine_reference": [],
        }
        with torch.no_grad():
            for start in range(0, len(indices), config.batch_size):
                batch = indices[start : start + config.batch_size]
                out = model(
                    tensors["composition"][batch].to(device),
                    tensors["foundation"][batch].to(device),
                    tensors["natoms"][batch].to(device),
                    tensors["nelements"][batch].to(device),
                )
                for key in chunks:
                    chunks[key].append(out[key].detach().cpu().numpy())
        return {key: np.concatenate(value) if value else np.asarray([], dtype=np.float32) for key, value in chunks.items()}

    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    best_epoch = -1
    history: list[dict[str, float | int]] = []
    stale = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        rng = np.random.default_rng(config.seed + epoch)
        rng.shuffle(train_indices)
        train_losses: list[float] = []
        for start in range(0, len(train_indices), config.batch_size):
            batch = train_indices[start : start + config.batch_size]
            out = model(
                tensors["composition"][batch].to(device),
                tensors["foundation"][batch].to(device),
                tensors["natoms"][batch].to(device),
                tensors["nelements"][batch].to(device),
            )
            yb = tensors["y_true"][batch].to(device)
            normalized_error = (out["pred"] - yb) / model.residual_std
            loss = loss_fn(normalized_error, torch.zeros_like(normalized_error))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
        scheduler.step()

        history_row: dict[str, float | int] = {"epoch": epoch, "train_loss": float(np.mean(train_losses))}
        if has_val:
            val_pred = predict_indices(val_indices)["pred"]
            val_mae = float(np.mean(np.abs(val_pred - y_true[val_indices])) * 1000.0)
            history_row["val_mae_mev_atom"] = val_mae
            if val_mae < best_val:
                best_val = val_mae
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
        history.append(history_row)
        if epoch % 10 == 0 or epoch == 1:
            if has_val:
                print(
                    f"[fegx0] epoch={epoch} train_loss={np.mean(train_losses):.5f} val_mae={val_mae:.3f}",
                    flush=True,
                )
            else:
                print(f"[fegx0] epoch={epoch} train_loss={np.mean(train_losses):.5f}", flush=True)
        if has_val and stale >= config.patience:
            print(f"[fegx0] early stop at epoch {epoch}; best epoch {best_epoch}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    all_indices = np.arange(len(frame))
    predictions = predict_indices(all_indices)
    frame["pred_element_reference_internal"] = predictions["element_reference"]
    frame["pred_affine_energy_composition_internal"] = predictions["affine_reference"]
    frame["fegx0_delta"] = predictions["adapter_delta"]
    frame["pred_fegx0"] = predictions["pred"]
    frame["fegx0_error"] = frame["pred_fegx0"] - frame["y_true"]
    frame["fegx0_abs_error_mev_atom"] = frame["fegx0_error"].abs() * 1000.0

    reference_diffs: dict[str, float] = {}
    if "pred_element_reference" in frame.columns:
        reference_diffs["max_abs_element_reference_init_diff_mev_atom"] = float(
            np.max(np.abs(element_ref - frame["pred_element_reference"].to_numpy(dtype=np.float32))) * 1000.0
        )
    if "pred_affine_energy_composition" in frame.columns:
        reference_diffs["max_abs_affine_reference_init_diff_mev_atom"] = float(
            np.max(np.abs(affine_ref - frame["pred_affine_energy_composition"].to_numpy(dtype=np.float32))) * 1000.0
        )

    metrics: dict[str, object] = {
        "config": {
            **config.__dict__,
            "device_used": str(device),
            "calibrated_predictions": str(calibrated_predictions),
            "feature_cache": str(feature_cache),
            "coefficient_csv": str(coefficient_csv),
            "best_epoch": best_epoch,
            "residual_mean": residual_mean,
            "residual_std": residual_std,
            "trainable_reference": trainable_reference,
            "feature_names": [*X_COLS, "model_energy_per_atom", "pred_element_reference_internal", "pred_affine_energy_composition_internal", "natoms", "nelements"],
            **reference_diffs,
        },
        "baseline": {},
        "fegx0": {},
    }
    for split, group in frame.groupby("split"):
        metrics["baseline"][split] = regression_metrics(
            group["y_true"].to_numpy(), group["pred_element_reference_internal"].to_numpy()
        )
        metrics["fegx0"][split] = {
            "overall": regression_metrics(group["y_true"].to_numpy(), group["pred_fegx0"].to_numpy()),
            "slices": slice_metrics(group, "pred_fegx0", "y_true"),
        }

    frame.to_csv(out_dir / "fegx0_predictions.csv", index=False)
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    torch.save(
        {
            "model": model.state_dict(),
            "config": config.__dict__,
            "coefficients": {
                key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in coeffs.items()
            },
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "residual_mean": residual_mean,
            "residual_std": residual_std,
        },
        out_dir / "fegx0.pt",
    )
    return metrics


def run_fegx1_training(
    *,
    calibrated_predictions: Path,
    feature_cache: Path,
    structure_cache: Path,
    coefficient_csv: Path,
    out_dir: Path,
    config: FEGX1Config,
) -> dict[str, object]:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = load_fegx1_frame(calibrated_predictions, feature_cache, structure_cache)
    frame["split"] = frame["split"].astype(str)
    train_mask = frame["split"].to_numpy() == "train"
    full_train_mask = train_mask.copy()
    val_mask = frame["split"].to_numpy() == "val"
    train_subset_summary: dict[str, object] = {
        "fraction": float(config.train_fraction),
        "seed": int(config.train_subset_seed),
        "full_train_rows": int(np.sum(full_train_mask)),
        "used_train_rows": int(np.sum(train_mask)),
        "enabled": False,
    }
    if config.train_fraction <= 0.0 or config.train_fraction > 1.0:
        raise ValueError("train_fraction must be in (0, 1]")
    if config.primary_train_mask_prob < 0.0 or config.primary_train_mask_prob > 1.0:
        raise ValueError("primary_train_mask_prob must be in [0, 1]")
    if config.primary_train_mask_prob > 0.0 and not config.use_primary_missing_branch:
        raise ValueError("primary_train_mask_prob requires use_primary_missing_branch=True")
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

    coeffs = load_reference_coefficients(coefficient_csv)
    x_comp = frame[X_COLS].to_numpy(dtype=np.float32)
    foundation = frame["model_energy_per_atom"].to_numpy(dtype=np.float32)
    natoms = frame["natoms"].to_numpy(dtype=np.float32)
    nelements = frame["nelements"].to_numpy(dtype=np.float32)
    y_true = frame["y_true"].to_numpy(dtype=np.float32)
    primary_missing = (
        pd.to_numeric(frame["primary_missing"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        if "primary_missing" in frame.columns
        else np.zeros(len(frame), dtype=np.float32)
    )
    if np.any(primary_missing > 0.5) and not config.use_primary_missing_branch:
        raise ValueError(
            "calibrated_predictions contains primary_missing rows but use_primary_missing_branch=False; "
            "enable the missing-aware branch for full-coverage training"
        )
    structure_scalars, spacegroup_index, crystal_index, scalar_mean, scalar_std, crystal_map = fegx1_prepare_structure(
        frame, train_mask
    )
    element_env_stats = np.zeros((len(frame), MAX_Z, len(ELEMENT_ENV_STATS)), dtype=np.float32)
    element_env_mean = np.zeros(len(ELEMENT_ENV_STATS), dtype=np.float32)
    element_env_std = np.ones(len(ELEMENT_ENV_STATS), dtype=np.float32)
    element_env_summary: dict[str, object] = {"enabled": False}
    if config.use_element_env:
        if not config.element_env_cache:
            raise ValueError("use_element_env=True requires --element-env-cache")
        element_env_stats, element_env_mean, element_env_std, element_env_summary = load_element_env_features(
            frame,
            Path(config.element_env_cache),
            train_mask,
        )
        element_env_summary["enabled"] = True
    pair_env_ids = np.zeros((len(frame), max(1, int(config.max_pair_env_pairs))), dtype=np.int64)
    pair_env_stats = np.zeros((len(frame), max(1, int(config.max_pair_env_pairs)), len(PAIR_ENV_STATS)), dtype=np.float32)
    pair_env_mean = np.zeros(len(PAIR_ENV_STATS), dtype=np.float32)
    pair_env_std = np.ones(len(PAIR_ENV_STATS), dtype=np.float32)
    pair_env_summary: dict[str, object] = {"enabled": False}
    if config.use_pair_env:
        if not config.pair_env_cache:
            raise ValueError("use_pair_env=True requires --pair-env-cache")
        pair_env_ids, pair_env_stats, pair_env_mean, pair_env_std, pair_env_summary = load_pair_env_features(
            frame,
            Path(config.pair_env_cache),
            train_mask,
            max_pairs=config.max_pair_env_pairs,
            ablation=config.pair_env_ablation,
        )
        pair_env_summary["enabled"] = True
    shell_pair_env_ids = np.zeros((len(frame), max(1, int(config.max_shell_pair_env_pairs))), dtype=np.int64)
    shell_pair_env_stats = np.zeros(
        (len(frame), max(1, int(config.max_shell_pair_env_pairs)), len(SHELL_PAIR_ENV_STATS)),
        dtype=np.float32,
    )
    shell_pair_env_mean = np.zeros(len(SHELL_PAIR_ENV_STATS), dtype=np.float32)
    shell_pair_env_std = np.ones(len(SHELL_PAIR_ENV_STATS), dtype=np.float32)
    shell_pair_env_summary: dict[str, object] = {"enabled": False}
    if config.use_shell_pair_env:
        if not config.shell_pair_env_cache:
            raise ValueError("use_shell_pair_env=True requires --shell-pair-env-cache")
        (
            shell_pair_env_ids,
            shell_pair_env_stats,
            shell_pair_env_mean,
            shell_pair_env_std,
            shell_pair_env_summary,
        ) = load_shell_pair_env_features(
            frame,
            Path(config.shell_pair_env_cache),
            train_mask,
            max_pairs=config.max_shell_pair_env_pairs,
        )
        shell_pair_env_summary["enabled"] = True
    chem_pair_env_ids = np.zeros((len(frame), max(1, int(config.max_chem_pair_env_pairs))), dtype=np.int64)
    chem_pair_env_stats = np.zeros(
        (len(frame), max(1, int(config.max_chem_pair_env_pairs)), len(CHEM_PAIR_ENV_STATS)),
        dtype=np.float32,
    )
    chem_pair_env_mean = np.zeros(len(CHEM_PAIR_ENV_STATS), dtype=np.float32)
    chem_pair_env_std = np.ones(len(CHEM_PAIR_ENV_STATS), dtype=np.float32)
    chem_pair_env_summary: dict[str, object] = {"enabled": False}
    if config.use_chem_pair_env:
        if not config.chem_pair_env_cache:
            raise ValueError("use_chem_pair_env=True requires --chem-pair-env-cache")
        (
            chem_pair_env_ids,
            chem_pair_env_stats,
            chem_pair_env_mean,
            chem_pair_env_std,
            chem_pair_env_summary,
        ) = load_chem_pair_env_features(
            frame,
            Path(config.chem_pair_env_cache),
            train_mask,
            max_pairs=config.max_chem_pair_env_pairs,
            mode=config.chem_pair_env_mode,
        )
        chem_pair_env_summary["enabled"] = True
    dual_foundation_stats = np.zeros((len(frame), len(DUAL_FOUNDATION_STATS)), dtype=np.float32)
    dual_foundation_mean = np.zeros(len(DUAL_FOUNDATION_STATS), dtype=np.float32)
    dual_foundation_std = np.ones(len(DUAL_FOUNDATION_STATS), dtype=np.float32)
    dual_foundation_summary: dict[str, object] = {"enabled": False}
    if config.use_dual_foundation:
        if not config.dual_foundation_calibrated_predictions:
            raise ValueError("use_dual_foundation=True requires --dual-foundation-calibrated-predictions")
        (
            dual_foundation_stats,
            dual_foundation_mean,
            dual_foundation_std,
            dual_foundation_summary,
        ) = load_dual_foundation_features(
            frame,
            Path(config.dual_foundation_calibrated_predictions),
            train_mask,
            ablation=config.dual_foundation_ablation,
        )
    tail_features = np.zeros((len(frame), 1), dtype=np.float32)
    tail_feature_mean = np.zeros(1, dtype=np.float32)
    tail_feature_std = np.ones(1, dtype=np.float32)
    tail_feature_cols: list[str] = []
    tail_feature_summary: dict[str, object] = {"enabled": False}
    if config.use_tail_features:
        if not config.tail_feature_cache:
            raise ValueError("use_tail_features=True requires --tail-feature-cache")
        (
            tail_features,
            tail_feature_mean,
            tail_feature_std,
            tail_feature_cols,
            tail_feature_summary,
        ) = load_tail_feature_cache(
            frame,
            Path(config.tail_feature_cache),
            train_mask,
            columns=config.tail_feature_cols,
        )

    element_ref = foundation - x_comp @ coeffs["element_mu"] - float(coeffs["element_bias"])
    residual = y_true - element_ref
    target_mean = float(y_true[train_mask].mean())
    target_std = float(y_true[train_mask].std())
    if target_std < 1e-8:
        target_std = 1.0
    residual_mean = float(residual[train_mask].mean())
    residual_std = float(residual[train_mask].std())
    if residual_std < 1e-8:
        residual_std = 1.0
    missing_reference_summary: dict[str, object] = {
        "enabled": bool(config.use_primary_missing_branch),
        "primary_train_mask_prob": float(config.primary_train_mask_prob),
        "primary_missing_branch_hidden_dim": int(config.primary_missing_branch_hidden_dim),
        "train_primary_missing_rows": int(np.sum(primary_missing[train_mask] > 0.5)),
        "val_primary_missing_rows": int(np.sum(primary_missing[val_mask] > 0.5)),
        "test_primary_missing_rows": int(np.sum(primary_missing[(frame["split"].to_numpy() == "test")] > 0.5)),
    }
    sample_weights, tail_weight_summary = build_tail_sample_weights(
        y_true=y_true,
        nelements=nelements,
        element_reference=element_ref,
        train_mask=train_mask,
        config=config,
    )
    rank_left, rank_right, rank_sign, rank_summary = build_formula_rank_pairs(
        frame,
        y_true=y_true,
        train_mask=train_mask,
        margin_mev=config.rank_margin_mev,
        pair_mode=config.rank_pair_mode,
        max_pairs_per_formula=config.rank_max_pairs_per_formula,
        seed=config.seed,
    )
    router_anchor_mask, router_anchor_risk_score, router_anchor_summary = build_router_anchor_mask(
        dual_foundation_stats=dual_foundation_stats,
        train_mask=train_mask,
        config=config,
    )

    device = torch.device(config.device if torch.cuda.is_available() and config.device.startswith("cuda") else "cpu")
    trainable_reference = config.reference_lr > 0.0
    model = FEGX1Model(
        element_mu=coeffs["element_mu"],
        element_bias=float(coeffs["element_bias"]),
        affine_alpha=float(coeffs["affine_alpha"]),
        affine_beta=coeffs["affine_beta"],
        affine_bias=float(coeffs["affine_bias"]),
        scalar_mean=scalar_mean,
        scalar_std=scalar_std,
        target_mean=target_mean,
        target_std=target_std,
        residual_mean=residual_mean,
        residual_std=residual_std,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        trainable_reference=trainable_reference,
        pair_dim=config.pair_dim,
        spacegroup_dim=config.spacegroup_dim,
        crystal_dim=config.crystal_dim,
        structure_hidden_dim=config.structure_hidden_dim,
        crystal_vocab_size=len(CRYSTAL_SYSTEMS),
        residual_head_type=config.residual_head_type,
        num_experts=config.num_experts,
        router_hidden_dim=config.router_hidden_dim,
        router_temperature=config.router_temperature,
        moe_router_init=config.moe_router_init,
        use_element_env=config.use_element_env,
        element_env_dim=config.element_env_dim,
        element_env_hidden_dim=config.element_env_hidden_dim,
        element_env_delta_weight=config.element_env_delta_weight,
        element_env_to_residual_head=config.element_env_to_residual_head,
        use_pair_env=config.use_pair_env,
        pair_env_dim=config.pair_env_dim,
        pair_env_hidden_dim=config.pair_env_hidden_dim,
        pair_env_delta_weight=config.pair_env_delta_weight,
        use_shell_pair_env=config.use_shell_pair_env,
        shell_pair_env_dim=config.shell_pair_env_dim,
        shell_pair_env_hidden_dim=config.shell_pair_env_hidden_dim,
        shell_pair_env_delta_weight=config.shell_pair_env_delta_weight,
        use_chem_pair_env=config.use_chem_pair_env,
        chem_pair_env_dim=config.chem_pair_env_dim,
        chem_pair_env_hidden_dim=config.chem_pair_env_hidden_dim,
        chem_pair_env_delta_weight=config.chem_pair_env_delta_weight,
        use_dual_foundation=config.use_dual_foundation,
        dual_foundation_dim=config.dual_foundation_dim,
        dual_foundation_hidden_dim=config.dual_foundation_hidden_dim,
        dual_foundation_mean=dual_foundation_mean,
        dual_foundation_std=dual_foundation_std,
        use_risk_residual_branch=config.use_risk_residual_branch,
        risk_branch_hidden_dim=config.risk_branch_hidden_dim,
        risk_branch_gate_bias=config.risk_branch_gate_bias,
        use_tail_features=config.use_tail_features,
        tail_feature_input_dim=tail_features.shape[1] if config.use_tail_features else 0,
        tail_feature_dim=config.tail_feature_dim,
        tail_feature_hidden_dim=config.tail_feature_hidden_dim,
        use_dual_reference_gate=config.use_dual_reference_gate,
        dual_reference_gate_hidden_dim=config.dual_reference_gate_hidden_dim,
        dual_reference_gate_init_weight=config.dual_reference_gate_init_weight,
        dual_reference_gate_use_raw_dual_stats=config.dual_reference_gate_use_raw_dual_stats,
        use_primary_missing_branch=config.use_primary_missing_branch,
        primary_missing_branch_hidden_dim=config.primary_missing_branch_hidden_dim,
        use_grouped_structure_adapter=config.use_grouped_structure_adapter,
        grouped_structure_group_dim=config.grouped_structure_group_dim,
        grouped_structure_group_hidden_dim=config.grouped_structure_group_hidden_dim,
        grouped_structure_context_hidden_dim=config.grouped_structure_context_hidden_dim,
        grouped_structure_context_dim=config.grouped_structure_context_dim,
        grouped_structure_interaction_dim=config.grouped_structure_interaction_dim,
        grouped_structure_dropout=config.grouped_structure_dropout,
        grouped_structure_magnitude_bias=config.grouped_structure_magnitude_bias,
        anchor_mode=config.anchor_mode,
        learnable_anchor_init=config.learnable_anchor_init,
    ).to(device)
    init_summary = initialize_fegx1_from_checkpoint(
        model,
        config.init_checkpoint,
        noise_std=config.moe_noise_std,
        transfer_mode=config.init_transfer_mode,
    )
    model = model.to(device)
    grouped_structure_freeze_summary: dict[str, object] = {"enabled": False}
    if config.use_grouped_structure_adapter and config.grouped_structure_freeze_base:
        trainable_prefixes = ("grouped_structure_adapter.",)
        for name, param in model.named_parameters():
            param.requires_grad = any(name.startswith(prefix) for prefix in trainable_prefixes)
        grouped_structure_freeze_summary = {
            "enabled": True,
            "trainable_prefixes": list(trainable_prefixes),
        }
    total_params = int(sum(param.numel() for param in model.parameters()))
    trainable_params = int(sum(param.numel() for param in model.parameters() if param.requires_grad))
    frozen_params = int(total_params - trainable_params)
    grouped_structure_freeze_summary.update(
        {
            "trainable_params": trainable_params,
            "frozen_params": frozen_params,
        }
    )

    ref_params = [
        param
        for name, param in model.named_parameters()
        if (name.startswith("element_reference.") or name.startswith("affine_reference.")) and param.requires_grad
    ]
    main_params = [
        param
        for name, param in model.named_parameters()
        if not (name.startswith("element_reference.") or name.startswith("affine_reference.")) and param.requires_grad
    ]
    param_groups: list[dict[str, object]] = [{"params": main_params, "lr": config.lr, "weight_decay": config.weight_decay}]
    if trainable_reference:
        param_groups.append({"params": ref_params, "lr": config.reference_lr, "weight_decay": 0.0})
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, config.epochs), eta_min=config.lr * 0.05)
    loss_fn = nn.SmoothL1Loss(beta=config.huber_beta)
    loss_fn_none = nn.SmoothL1Loss(beta=config.huber_beta, reduction="none")
    bce_loss = nn.BCEWithLogitsLoss()

    tensors = {
        "composition": torch.from_numpy(x_comp),
        "foundation": torch.from_numpy(foundation),
        "natoms": torch.from_numpy(natoms),
        "nelements": torch.from_numpy(nelements),
        "structure_scalars": torch.from_numpy(structure_scalars),
        "spacegroup_index": torch.from_numpy(spacegroup_index),
        "crystal_index": torch.from_numpy(crystal_index),
        "element_env_stats": torch.from_numpy(element_env_stats),
        "pair_env_ids": torch.from_numpy(pair_env_ids),
        "pair_env_stats": torch.from_numpy(pair_env_stats),
        "shell_pair_env_ids": torch.from_numpy(shell_pair_env_ids),
        "shell_pair_env_stats": torch.from_numpy(shell_pair_env_stats),
        "chem_pair_env_ids": torch.from_numpy(chem_pair_env_ids),
        "chem_pair_env_stats": torch.from_numpy(chem_pair_env_stats),
        "dual_foundation_stats": torch.from_numpy(dual_foundation_stats),
        "tail_features": torch.from_numpy(tail_features),
        "primary_missing": torch.from_numpy(primary_missing),
        "router_anchor_mask": torch.from_numpy(router_anchor_mask),
        "router_anchor_risk_score": torch.from_numpy(router_anchor_risk_score),
        "y_true": torch.from_numpy(y_true),
        "sample_weights": torch.from_numpy(sample_weights),
        "rank_left": torch.from_numpy(rank_left),
        "rank_right": torch.from_numpy(rank_right),
        "rank_sign": torch.from_numpy(rank_sign),
    }
    train_indices = np.flatnonzero(train_mask)
    val_indices = np.flatnonzero(val_mask)
    has_val = len(val_indices) > 0
    train_batches_preview, train_sampler_summary = build_train_batches(
        frame,
        train_indices=train_indices,
        batch_size=int(config.batch_size),
        seed=int(config.seed),
        epoch=0,
        sampling_mode=config.train_sampling_mode,
        hybrid_primary_ratio=float(config.hybrid_sampling_primary_ratio),
        hybrid_group_col=config.hybrid_sampling_group_col,
    )

    def forward_batch(
        batch: np.ndarray,
        *,
        primary_missing_override: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        primary_missing_batch = (
            primary_missing_override
            if primary_missing_override is not None
            else tensors["primary_missing"][batch].to(device)
        )
        dual_stats_batch = None
        if config.use_dual_foundation:
            dual_stats_batch = tensors["dual_foundation_stats"][batch].to(device)
            dual_stats_batch = apply_primary_missing_to_dual_stats_tensor(dual_stats_batch, primary_missing_batch)
        return model(
            composition=tensors["composition"][batch].to(device),
            foundation_energy=tensors["foundation"][batch].to(device),
            natoms=tensors["natoms"][batch].to(device),
            nelements=tensors["nelements"][batch].to(device),
            structure_scalars=tensors["structure_scalars"][batch].to(device),
            spacegroup_index=tensors["spacegroup_index"][batch].to(device),
            crystal_index=tensors["crystal_index"][batch].to(device),
            element_env_stats=tensors["element_env_stats"][batch].to(device) if config.use_element_env else None,
            pair_env_ids=tensors["pair_env_ids"][batch].to(device) if config.use_pair_env else None,
            pair_env_stats=tensors["pair_env_stats"][batch].to(device) if config.use_pair_env else None,
            shell_pair_env_ids=tensors["shell_pair_env_ids"][batch].to(device) if config.use_shell_pair_env else None,
            shell_pair_env_stats=tensors["shell_pair_env_stats"][batch].to(device) if config.use_shell_pair_env else None,
            chem_pair_env_ids=tensors["chem_pair_env_ids"][batch].to(device) if config.use_chem_pair_env else None,
            chem_pair_env_stats=tensors["chem_pair_env_stats"][batch].to(device) if config.use_chem_pair_env else None,
            dual_foundation_stats=dual_stats_batch,
            tail_features=tensors["tail_features"][batch].to(device) if config.use_tail_features else None,
            primary_missing=primary_missing_batch,
        )

    def ranking_loss_for_pair_rows(pair_rows: np.ndarray) -> torch.Tensor:
        if len(pair_rows) == 0:
            return torch.zeros((), device=device)
        left_batch = rank_left[pair_rows]
        right_batch = rank_right[pair_rows]
        signs = torch.from_numpy(rank_sign[pair_rows]).to(device)
        left_out = forward_batch(left_batch)
        right_out = forward_batch(right_batch)
        pred_delta = right_out["pred"] - left_out["pred"]
        tau = max(float(config.rank_temperature), 1e-6)
        return F.softplus(-signs * pred_delta / tau).mean()

    def predict_indices(indices: np.ndarray) -> dict[str, np.ndarray]:
        model.eval()
        chunks: dict[str, list[np.ndarray]] = {
            "pred": [],
            "base_pred": [],
            "adapter_delta": [],
            "anchor_component": [],
            "anchor_alpha": [],
            "grouped_structure_delta": [],
            "grouped_structure_raw_delta": [],
            "grouped_structure_magnitude": [],
            "grouped_structure_weights": [],
            "element_reference": [],
            "dual_reference": [],
            "aux_element_reference": [],
            "dual_reference_gate": [],
            "affine_reference": [],
            "log_sigma": [],
            "positive_logit": [],
            "near_zero_logit": [],
            "router_weights": [],
            "env_delta": [],
            "pair_env_delta": [],
            "shell_pair_env_delta": [],
            "chem_pair_env_delta": [],
            "risk_residual_delta": [],
            "risk_branch_gate": [],
        }
        with torch.no_grad():
            for start in range(0, len(indices), config.batch_size):
                batch = indices[start : start + config.batch_size]
                out = forward_batch(batch)
                for key in chunks:
                    if key == "router_weights" and out[key].numel() == 0:
                        continue
                    chunks[key].append(out[key].detach().cpu().numpy())
        return {key: np.concatenate(value) if value else np.asarray([], dtype=np.float32) for key, value in chunks.items()}

    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    best_epoch = -1
    history: list[dict[str, float | int]] = []
    stale = 0
    train_steps_per_epoch = max(1, int(len(train_batches_preview)))
    rank_rows_per_step = 0
    if config.rank_weight > 0.0 and len(rank_left):
        rank_rows_per_step = min(
            int(config.rank_batch_size),
            max(1, int(np.ceil(max(1, config.rank_pairs_per_epoch) / train_steps_per_epoch))),
        )

    for epoch in range(1, config.epochs + 1):
        model.train()
        rng = np.random.default_rng(config.seed + epoch)
        train_batches, train_sampler_summary = build_train_batches(
            frame,
            train_indices=train_indices,
            batch_size=int(config.batch_size),
            seed=int(config.seed),
            epoch=epoch,
            sampling_mode=config.train_sampling_mode,
            hybrid_primary_ratio=float(config.hybrid_sampling_primary_ratio),
            hybrid_group_col=config.hybrid_sampling_group_col,
        )
        train_losses: list[float] = []
        train_main_losses: list[float] = []
        train_risk_losses: list[float] = []
        train_pos_losses: list[float] = []
        train_near_losses: list[float] = []
        train_rank_losses: list[float] = []
        train_tail_l2_losses: list[float] = []
        train_moe_lb_losses: list[float] = []
        train_router_anchor_losses: list[float] = []
        train_risk_branch_anchor_losses: list[float] = []
        train_dual_reference_gate_prior_losses: list[float] = []
        train_dual_reference_gate_entropy_values: list[float] = []
        train_moe_entropy_values: list[float] = []
        train_moe_max_usage_values: list[float] = []
        train_primary_masked_fractions: list[float] = []
        for batch in train_batches:
            primary_missing_batch = tensors["primary_missing"][batch].to(device)
            masked_primary_missing = primary_missing_batch
            masked_fraction = 0.0
            if config.use_primary_missing_branch and config.primary_train_mask_prob > 0.0:
                available_mask = primary_missing_batch < 0.5
                if bool(torch.any(available_mask).detach().cpu()):
                    sampled = torch.rand(primary_missing_batch.shape[0], device=device) < float(config.primary_train_mask_prob)
                    sampled = sampled & available_mask
                    masked_primary_missing = torch.where(sampled, torch.ones_like(primary_missing_batch), primary_missing_batch)
                    masked_fraction = float(torch.mean(sampled.float()).detach().cpu())
            out = forward_batch(batch, primary_missing_override=masked_primary_missing)
            yb = tensors["y_true"][batch].to(device)
            normalized_error = (out["pred"] - yb) / model.residual_std
            if tail_weight_summary["enabled"]:
                wb = tensors["sample_weights"][batch].to(device)
                per_sample = loss_fn_none(normalized_error, torch.zeros_like(normalized_error))
                main_loss = torch.sum(per_sample * wb) / torch.clamp(torch.sum(wb), min=1e-6)
            else:
                main_loss = loss_fn(normalized_error, torch.zeros_like(normalized_error))
            loss = main_loss

            tail_l2_loss_value = torch.zeros((), device=device)
            if config.tail_l2_weight > 0.0 and tail_weight_summary["enabled"]:
                wb = tensors["sample_weights"][batch].to(device)
                excess = torch.clamp(wb - 1.0, min=0.0)
                if float(torch.sum(excess).detach().cpu()) > 0.0:
                    tail_l2_loss_value = torch.sum(excess * normalized_error.square()) / torch.clamp(torch.sum(excess), min=1e-6)
                    loss = loss + config.tail_l2_weight * tail_l2_loss_value

            risk_loss_value = torch.zeros((), device=device)
            if config.risk_weight > 0.0:
                if config.risk_loss == "heteroscedastic":
                    log_sigma = out["log_sigma"].clamp(-4.0, 4.0)
                    risk_loss_value = torch.mean(
                        torch.abs(out["pred"] - yb) / torch.exp(log_sigma) + 0.05 * log_sigma
                    )
                else:
                    with torch.no_grad():
                        target_abs = torch.abs(normalized_error)
                    risk_pred = F.softplus(out["log_sigma"])
                    risk_loss_value = loss_fn(risk_pred, target_abs)
                loss = loss + config.risk_weight * risk_loss_value

            pos_loss_value = torch.zeros((), device=device)
            if config.pos_weight > 0.0:
                pos_target = (yb >= 0.0).float()
                pos_loss_value = bce_loss(out["positive_logit"], pos_target)
                loss = loss + config.pos_weight * pos_loss_value

            near_loss_value = torch.zeros((), device=device)
            if config.near_weight > 0.0:
                near_target = (torch.abs(yb) <= config.near_zero_threshold).float()
                near_loss_value = bce_loss(out["near_zero_logit"], near_target)
                loss = loss + config.near_weight * near_loss_value

            rank_loss_value = torch.zeros((), device=device)
            if config.rank_weight > 0.0 and len(rank_left):
                n_rank = min(rank_rows_per_step, len(rank_left))
                rank_rows = rng.choice(len(rank_left), size=n_rank, replace=False if n_rank < len(rank_left) else True)
                rank_loss_value = ranking_loss_for_pair_rows(np.asarray(rank_rows, dtype=np.int64))
                loss = loss + config.rank_weight * rank_loss_value

            moe_lb_loss_value = torch.zeros((), device=device)
            moe_entropy_value = torch.zeros((), device=device)
            moe_max_usage_value = torch.zeros((), device=device)
            if config.residual_head_type == "moe":
                moe_lb_loss_value, moe_entropy_value, moe_max_usage_value = moe_regularization(out["router_weights"])
                if config.moe_load_balance_weight > 0.0:
                    loss = loss + config.moe_load_balance_weight * moe_lb_loss_value
                if config.moe_entropy_weight != 0.0:
                    loss = loss - config.moe_entropy_weight * moe_entropy_value

            router_anchor_loss_value = torch.zeros((), device=device)
            if router_anchor_summary["enabled"]:
                anchor_mask = tensors["router_anchor_mask"][batch].to(device)
                if bool(torch.any(anchor_mask).detach().cpu()):
                    anchor_weights = out["router_weights"][anchor_mask]
                    target = torch.full(
                        (anchor_weights.shape[0],),
                        int(config.router_anchor_target_expert),
                        dtype=torch.long,
                        device=device,
                    )
                    router_anchor_loss_value = F.nll_loss(torch.log(anchor_weights.clamp_min(1e-8)), target)
                    loss = loss + config.router_anchor_weight * router_anchor_loss_value

            risk_branch_anchor_loss_value = torch.zeros((), device=device)
            if (
                config.use_risk_residual_branch
                and config.risk_branch_anchor_weight > 0.0
                and router_anchor_summary["enabled"]
            ):
                anchor_mask = tensors["router_anchor_mask"][batch].to(device)
                if bool(torch.any(anchor_mask).detach().cpu()):
                    gate_target = anchor_mask.float()
                    risk_branch_anchor_loss_value = bce_loss(out["risk_branch_gate_logit"], gate_target)
                    loss = loss + config.risk_branch_anchor_weight * risk_branch_anchor_loss_value

            gate_prior_loss_value = torch.zeros((), device=device)
            gate_entropy_value = torch.zeros((), device=device)
            if config.use_dual_reference_gate:
                gate = out["dual_reference_gate"].clamp(1e-6, 1.0 - 1e-6)
                gate_prior_loss_value = torch.abs(gate.mean() - float(config.dual_reference_gate_init_weight))
                gate_entropy_value = -torch.mean(gate * torch.log(gate) + (1.0 - gate) * torch.log(1.0 - gate))
                if config.dual_reference_gate_prior_weight > 0.0:
                    loss = loss + config.dual_reference_gate_prior_weight * gate_prior_loss_value
                if config.dual_reference_gate_entropy_weight != 0.0:
                    loss = loss - config.dual_reference_gate_entropy_weight * gate_entropy_value

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))
            train_main_losses.append(float(main_loss.detach().cpu()))
            train_risk_losses.append(float(risk_loss_value.detach().cpu()))
            train_pos_losses.append(float(pos_loss_value.detach().cpu()))
            train_near_losses.append(float(near_loss_value.detach().cpu()))
            train_rank_losses.append(float(rank_loss_value.detach().cpu()))
            train_tail_l2_losses.append(float(tail_l2_loss_value.detach().cpu()))
            train_moe_lb_losses.append(float(moe_lb_loss_value.detach().cpu()))
            train_router_anchor_losses.append(float(router_anchor_loss_value.detach().cpu()))
            train_risk_branch_anchor_losses.append(float(risk_branch_anchor_loss_value.detach().cpu()))
            train_dual_reference_gate_prior_losses.append(float(gate_prior_loss_value.detach().cpu()))
            train_dual_reference_gate_entropy_values.append(float(gate_entropy_value.detach().cpu()))
            train_moe_entropy_values.append(float(moe_entropy_value.detach().cpu()))
            train_moe_max_usage_values.append(float(moe_max_usage_value.detach().cpu()))
            train_primary_masked_fractions.append(masked_fraction)
        scheduler.step()

        history_row: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "train_main_loss": float(np.mean(train_main_losses)),
            "train_risk_loss": float(np.mean(train_risk_losses)),
            "train_pos_loss": float(np.mean(train_pos_losses)),
            "train_near_loss": float(np.mean(train_near_losses)),
            "train_rank_loss": float(np.mean(train_rank_losses)),
            "train_tail_l2_loss": float(np.mean(train_tail_l2_losses)),
            "train_moe_load_balance_loss": float(np.mean(train_moe_lb_losses)),
            "train_router_anchor_loss": float(np.mean(train_router_anchor_losses)),
            "train_risk_branch_anchor_loss": float(np.mean(train_risk_branch_anchor_losses)),
            "train_dual_reference_gate_prior_loss": float(np.mean(train_dual_reference_gate_prior_losses)),
            "train_dual_reference_gate_entropy": float(np.mean(train_dual_reference_gate_entropy_values)),
            "train_moe_entropy": float(np.mean(train_moe_entropy_values)),
            "train_moe_max_usage": float(np.mean(train_moe_max_usage_values)),
            "train_primary_masked_fraction": float(np.mean(train_primary_masked_fractions)) if train_primary_masked_fractions else 0.0,
        }
        extra_val_bits: list[str] = []
        if has_val:
            val_pred = predict_indices(val_indices)["pred"]
            val_mae = float(np.mean(np.abs(val_pred - y_true[val_indices])) * 1000.0)
            history_row["val_mae_mev_atom"] = val_mae
            if config.validation_bucket_col and config.validation_bucket_col in frame.columns and len(val_indices):
                val_buckets = frame.iloc[val_indices][config.validation_bucket_col].astype(str).fillna("").to_numpy()
                for bucket_name in sorted({name.strip() for name in val_buckets if str(name).strip()}):
                    mask = val_buckets == bucket_name
                    if not bool(np.any(mask)):
                        continue
                    bucket_mae = float(np.mean(np.abs(val_pred[mask] - y_true[val_indices][mask])) * 1000.0)
                    key = f"val_{_metric_slug(bucket_name)}_mae_mev_atom"
                    history_row[key] = bucket_mae
                    extra_val_bits.append(f"{_metric_slug(bucket_name)}={bucket_mae:.3f}")
        history.append(history_row)
        if has_val:
            if val_mae < best_val:
                best_val = val_mae
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
        if epoch % 10 == 0 or epoch == 1:
            if has_val:
                print(
                    f"[{config.exp_name}] epoch={epoch} train_loss={np.mean(train_losses):.5f} "
                    f"val_mae={val_mae:.3f}"
                    + (f" ({', '.join(extra_val_bits)})" if extra_val_bits else ""),
                    flush=True,
                )
            else:
                print(f"[{config.exp_name}] epoch={epoch} train_loss={np.mean(train_losses):.5f}", flush=True)
        if has_val and stale >= config.patience:
            print(f"[{config.exp_name}] early stop at epoch {epoch}; best epoch {best_epoch}", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    all_indices = np.arange(len(frame))
    predictions = predict_indices(all_indices)
    pred_col = f"pred_{config.exp_name}"
    frame["pred_element_reference_internal"] = predictions["element_reference"]
    frame["pred_dual_reference_internal"] = predictions["dual_reference"] if predictions["dual_reference"].size else predictions["element_reference"]
    frame["pred_aux_element_reference_internal"] = (
        predictions["aux_element_reference"] if predictions["aux_element_reference"].size else 0.0
    )
    frame[f"{config.exp_name}_dual_reference_gate"] = (
        predictions["dual_reference_gate"] if predictions["dual_reference_gate"].size else 1.0
    )
    frame["pred_affine_energy_composition_internal"] = predictions["affine_reference"]
    frame[f"{config.exp_name}_base_pred"] = predictions["base_pred"] if predictions["base_pred"].size else 0.0
    frame[f"{config.exp_name}_delta"] = predictions["adapter_delta"]
    frame[f"{config.exp_name}_grouped_structure_delta"] = (
        predictions["grouped_structure_delta"] if predictions["grouped_structure_delta"].size else 0.0
    )
    frame[f"{config.exp_name}_grouped_structure_raw_delta"] = (
        predictions["grouped_structure_raw_delta"] if predictions["grouped_structure_raw_delta"].size else 0.0
    )
    frame[f"{config.exp_name}_grouped_structure_magnitude"] = (
        predictions["grouped_structure_magnitude"] if predictions["grouped_structure_magnitude"].size else 0.0
    )
    frame[f"{config.exp_name}_env_delta"] = predictions["env_delta"] if predictions["env_delta"].size else 0.0
    frame[f"{config.exp_name}_pair_env_delta"] = (
        predictions["pair_env_delta"] if predictions["pair_env_delta"].size else 0.0
    )
    frame[f"{config.exp_name}_shell_pair_env_delta"] = (
        predictions["shell_pair_env_delta"] if predictions["shell_pair_env_delta"].size else 0.0
    )
    frame[f"{config.exp_name}_chem_pair_env_delta"] = (
        predictions["chem_pair_env_delta"] if predictions["chem_pair_env_delta"].size else 0.0
    )
    frame[f"{config.exp_name}_risk_residual_delta"] = (
        predictions["risk_residual_delta"] if predictions["risk_residual_delta"].size else 0.0
    )
    frame[f"{config.exp_name}_risk_branch_gate"] = (
        predictions["risk_branch_gate"] if predictions["risk_branch_gate"].size else 0.0
    )
    if predictions["anchor_alpha"].size:
        frame[f"{config.exp_name}_anchor_alpha"] = predictions["anchor_alpha"]
    frame[pred_col] = predictions["pred"]
    frame[f"{config.exp_name}_error"] = frame[pred_col] - frame["y_true"]
    frame[f"{config.exp_name}_abs_error_mev_atom"] = frame[f"{config.exp_name}_error"].abs() * 1000.0
    frame[f"{config.exp_name}_log_sigma"] = predictions["log_sigma"]
    frame[f"{config.exp_name}_positive_logit"] = predictions["positive_logit"]
    frame[f"{config.exp_name}_near_zero_logit"] = predictions["near_zero_logit"]
    frame["primary_missing"] = primary_missing.astype(np.float32)
    if config.use_dual_foundation:
        aux_missing_idx = DUAL_FOUNDATION_STATS.index("aux_missing")
        frame["aux_missing"] = dual_foundation_stats[:, aux_missing_idx].astype(np.float32)
    else:
        frame["aux_missing"] = np.ones(len(frame), dtype=np.float32)
    frame["prediction_path"] = np.where(
        frame["primary_missing"].to_numpy(dtype=np.float32) > 0.5,
        "no_primary_path",
        np.where(
            (config.use_dual_foundation) & (frame["aux_missing"].to_numpy(dtype=np.float32) < 0.5),
            "dual_reference_path",
            "single_reference_path",
        ),
    )
    path_summary = {
        str(split): {
            str(path): int(count)
            for path, count in group["prediction_path"].value_counts(dropna=False).sort_index().items()
        }
        for split, group in frame.groupby("split")
    }
    missing_reference_summary["prediction_paths"] = path_summary
    router_summary: dict[str, object] = {"enabled": config.residual_head_type == "moe"}
    if predictions["router_weights"].size:
        router_weights = predictions["router_weights"]
        for idx in range(router_weights.shape[1]):
            frame[f"{config.exp_name}_router_w{idx}"] = router_weights[:, idx]
        test_router = router_weights[frame["split"].to_numpy() == "test"]
        train_router = router_weights[train_mask]
        router_summary.update(
            {
                "num_experts": int(router_weights.shape[1]),
                "train_mean_usage": train_router.mean(axis=0).tolist() if len(train_router) else [],
                "test_mean_usage": test_router.mean(axis=0).tolist() if len(test_router) else [],
                "test_max_mean_usage": float(test_router.mean(axis=0).max()) if len(test_router) else float("nan"),
                "test_mean_entropy": float(
                    np.mean(-np.sum(test_router * np.log(np.clip(test_router, 1e-8, 1.0)), axis=1))
                )
                if len(test_router)
                else float("nan"),
            }
        )
    grouped_structure_summary: dict[str, object] = {
        "enabled": bool(config.use_grouped_structure_adapter),
        "freeze_base": bool(config.grouped_structure_freeze_base),
        "group_names": list(GROUPED_STRUCTURE_GROUP_NAMES),
        "group_dim": int(config.grouped_structure_group_dim),
        "group_hidden_dim": int(config.grouped_structure_group_hidden_dim),
        "context_hidden_dim": int(config.grouped_structure_context_hidden_dim),
        "context_dim": int(config.grouped_structure_context_dim),
        "interaction_dim": int(config.grouped_structure_interaction_dim),
        "dropout": float(config.grouped_structure_dropout),
        "magnitude_bias": float(config.grouped_structure_magnitude_bias),
        "freeze_summary": grouped_structure_freeze_summary,
    }
    if config.use_grouped_structure_adapter and predictions["grouped_structure_weights"].size:
        weights = predictions["grouped_structure_weights"]
        grouped_delta = predictions["grouped_structure_delta"]
        grouped_raw_delta = predictions["grouped_structure_raw_delta"]
        grouped_magnitude = predictions["grouped_structure_magnitude"]
        for idx, name in enumerate(GROUPED_STRUCTURE_GROUP_NAMES):
            frame[f"{config.exp_name}_struct_weight_{name}"] = weights[:, idx]
        test_weights = weights[frame["split"].to_numpy() == "test"]
        train_weights = weights[train_mask]
        test_delta = grouped_delta[frame["split"].to_numpy() == "test"]
        train_delta = grouped_delta[train_mask]
        test_raw_delta = grouped_raw_delta[frame["split"].to_numpy() == "test"]
        train_raw_delta = grouped_raw_delta[train_mask]
        test_magnitude = grouped_magnitude[frame["split"].to_numpy() == "test"]
        train_magnitude = grouped_magnitude[train_mask]
        grouped_structure_summary.update(
            {
                "train_mean_weights": {
                    name: float(train_weights[:, idx].mean()) if len(train_weights) else float("nan")
                    for idx, name in enumerate(GROUPED_STRUCTURE_GROUP_NAMES)
                },
                "test_mean_weights": {
                    name: float(test_weights[:, idx].mean()) if len(test_weights) else float("nan")
                    for idx, name in enumerate(GROUPED_STRUCTURE_GROUP_NAMES)
                },
                "train_active_fraction": {
                    name: float(np.mean(train_weights[:, idx] > 1e-6)) if len(train_weights) else float("nan")
                    for idx, name in enumerate(GROUPED_STRUCTURE_GROUP_NAMES)
                },
                "test_active_fraction": {
                    name: float(np.mean(test_weights[:, idx] > 1e-6)) if len(test_weights) else float("nan")
                    for idx, name in enumerate(GROUPED_STRUCTURE_GROUP_NAMES)
                },
                "train_mean_magnitude": float(np.mean(train_magnitude)) if len(train_magnitude) else float("nan"),
                "test_mean_magnitude": float(np.mean(test_magnitude)) if len(test_magnitude) else float("nan"),
                "train_mean_abs_delta_mev_atom": float(np.mean(np.abs(train_delta)) * 1000.0) if len(train_delta) else float("nan"),
                "test_mean_abs_delta_mev_atom": float(np.mean(np.abs(test_delta)) * 1000.0) if len(test_delta) else float("nan"),
                "train_mean_abs_raw_delta_mev_atom": float(np.mean(np.abs(train_raw_delta)) * 1000.0)
                if len(train_raw_delta)
                else float("nan"),
                "test_mean_abs_raw_delta_mev_atom": float(np.mean(np.abs(test_raw_delta)) * 1000.0)
                if len(test_raw_delta)
                else float("nan"),
            }
        )
    final_anchor_alpha = float(predictions["anchor_alpha"][0]) if predictions["anchor_alpha"].size else None
    anchor_summary: dict[str, object] = {
        "mode": str(config.anchor_mode),
        "learnable_init": float(config.learnable_anchor_init),
        "trainable": bool(config.anchor_mode == "learnable_anchor"),
        "final_alpha": final_anchor_alpha,
    }
    dual_reference_gate_summary: dict[str, object] = {"enabled": bool(config.use_dual_reference_gate)}
    if predictions["dual_reference_gate"].size:
        gate = predictions["dual_reference_gate"]
        test_gate = gate[frame["split"].to_numpy() == "test"]
        train_gate = gate[train_mask]
        dual_reference_gate_summary.update(
            {
                "init_weight": float(config.dual_reference_gate_init_weight),
                "hidden_dim": int(config.dual_reference_gate_hidden_dim),
                "prior_weight": float(config.dual_reference_gate_prior_weight),
                "entropy_weight": float(config.dual_reference_gate_entropy_weight),
                "use_raw_dual_stats": bool(config.dual_reference_gate_use_raw_dual_stats),
                "train_mean": float(np.mean(train_gate)) if len(train_gate) else float("nan"),
                "train_std": float(np.std(train_gate)) if len(train_gate) else float("nan"),
                "test_mean": float(np.mean(test_gate)) if len(test_gate) else float("nan"),
                "test_std": float(np.std(test_gate)) if len(test_gate) else float("nan"),
                "test_p05": float(np.quantile(test_gate, 0.05)) if len(test_gate) else float("nan"),
                "test_p95": float(np.quantile(test_gate, 0.95)) if len(test_gate) else float("nan"),
            }
        )

    model_metrics = fegx1_metrics(frame, pred_col)
    if config.hybrid_sampling_group_col and config.hybrid_sampling_group_col in frame.columns:
        for split, group in frame.groupby("split"):
            model_metrics[str(split)]["chemistry_groups"] = grouped_regression_metrics(
                group,
                pred_col=pred_col,
                target_col="y_true",
                group_col=config.hybrid_sampling_group_col,
            )
    if config.validation_bucket_col and config.validation_bucket_col in frame.columns:
        val_group = frame.loc[frame["split"].astype(str) == "val"]
        if len(val_group):
            model_metrics["val"]["validation_buckets"] = grouped_regression_metrics(
                val_group,
                pred_col=pred_col,
                target_col="y_true",
                group_col=config.validation_bucket_col,
            )

    metrics: dict[str, object] = {
        "config": {
            **config.__dict__,
            "device_used": str(device),
            "calibrated_predictions": str(calibrated_predictions),
            "feature_cache": str(feature_cache),
            "structure_cache": str(structure_cache),
            "coefficient_csv": str(coefficient_csv),
            "best_epoch": best_epoch,
            "target_mean": target_mean,
            "target_std": target_std,
            "residual_mean": residual_mean,
            "residual_std": residual_std,
            "trainable_reference": trainable_reference,
            "trainable_params": trainable_params,
            "total_params": total_params,
            "frozen_params": frozen_params,
            "structure_scalar_cols": STRUCTURE_SCALAR_COLS,
            "crystal_map": crystal_map,
            "train_subset": train_subset_summary,
            "train_sampler": train_sampler_summary,
            "ranking": rank_summary,
            "ranking_pairs_per_step": int(rank_rows_per_step),
            "tail_weights": tail_weight_summary,
            "checkpoint_initialization": init_summary,
            "router": router_summary,
            "element_env": element_env_summary,
            "pair_env": pair_env_summary,
            "shell_pair_env": shell_pair_env_summary,
            "chem_pair_env": chem_pair_env_summary,
            "tail_features": tail_feature_summary,
            "dual_foundation": dual_foundation_summary,
            "missing_reference": missing_reference_summary,
            "router_anchor": router_anchor_summary,
            "dual_reference_gate": dual_reference_gate_summary,
            "anchor": anchor_summary,
            "grouped_structure": grouped_structure_summary,
            "risk_residual_branch": {
                "enabled": bool(config.use_risk_residual_branch),
                "hidden_dim": int(config.risk_branch_hidden_dim),
                "gate_bias": float(config.risk_branch_gate_bias),
                "anchor_weight": float(config.risk_branch_anchor_weight),
            },
        },
        "baseline": {},
        config.exp_name: model_metrics,
    }
    for split, group in frame.groupby("split"):
        metrics["baseline"][split] = regression_metrics(
            group["y_true"].to_numpy(), group["pred_element_reference_internal"].to_numpy()
        )

    out_pred = f"{config.exp_name}_predictions.csv"
    frame.to_csv(out_dir / out_pred, index=False)
    pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
    torch.save(
        {
            "model": model.state_dict(),
            "config": config.__dict__,
            "coefficients": {
                key: value.tolist() if isinstance(value, np.ndarray) else value for key, value in coeffs.items()
            },
            "scalar_mean": scalar_mean,
            "scalar_std": scalar_std,
            "element_env_mean": element_env_mean,
            "element_env_std": element_env_std,
            "pair_env_mean": pair_env_mean,
            "pair_env_std": pair_env_std,
            "shell_pair_env_mean": shell_pair_env_mean,
            "shell_pair_env_std": shell_pair_env_std,
            "chem_pair_env_mean": chem_pair_env_mean,
            "chem_pair_env_std": chem_pair_env_std,
            "tail_feature_mean": tail_feature_mean,
            "tail_feature_std": tail_feature_std,
            "tail_feature_cols": tail_feature_cols,
            "dual_foundation_mean": dual_foundation_mean,
            "dual_foundation_std": dual_foundation_std,
            "target_mean": target_mean,
            "target_std": target_std,
            "residual_mean": residual_mean,
            "residual_std": residual_std,
            "structure_scalar_cols": STRUCTURE_SCALAR_COLS,
            "crystal_map": crystal_map,
            "trainable_params": trainable_params,
            "total_params": total_params,
            "frozen_params": frozen_params,
            "train_subset": train_subset_summary,
            "train_sampler": train_sampler_summary,
            "checkpoint_initialization": init_summary,
            "element_env": element_env_summary,
            "pair_env": pair_env_summary,
            "shell_pair_env": shell_pair_env_summary,
            "chem_pair_env": chem_pair_env_summary,
            "tail_features": tail_feature_summary,
            "dual_foundation": dual_foundation_summary,
            "missing_reference": missing_reference_summary,
            "router_anchor": router_anchor_summary,
            "dual_reference_gate": dual_reference_gate_summary,
            "anchor": anchor_summary,
            "grouped_structure": grouped_structure_summary,
            "risk_residual_branch": {
                "enabled": bool(config.use_risk_residual_branch),
                "hidden_dim": int(config.risk_branch_hidden_dim),
                "gate_bias": float(config.risk_branch_gate_bias),
                "anchor_weight": float(config.risk_branch_anchor_weight),
            },
        },
        out_dir / f"{config.exp_name}.pt",
    )
    return metrics
