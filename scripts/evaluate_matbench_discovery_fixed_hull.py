#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import types
import zipfile

import numpy as np
import pandas as pd


THRESHOLDS = [0.0, 0.025, 0.05]
TOP_K = [20, 50, 100, 250, 500, 1000]
DEFAULT_OFFICIAL_WHEEL = Path(
    "<PROJECT_ROOT>/external_datasets/matbench_discovery_meta/"
    "matbench_discovery-1.3.1-py2.py3-none-any.whl"
)


def classify_stable(each_true: np.ndarray, each_pred: np.ndarray, threshold: float):
    actual_pos = each_true <= threshold
    actual_neg = each_true > threshold
    model_pos = each_pred <= threshold
    model_neg = each_pred > threshold
    nan_mask = ~np.isfinite(each_pred)
    if np.any(nan_mask):
        model_pos[nan_mask] = False
        model_neg[nan_mask] = True
    return actual_pos & model_pos, actual_pos & model_neg, actual_neg & model_pos, actual_neg & model_neg


def safe_div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[ok]
    y_pred = y_pred[ok]
    if len(y_true) == 0:
        return float("nan")
    denom = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - float(np.sum((y_pred - y_true) ** 2)) / denom if denom > 0 else float("nan")


def stable_metrics(each_true, each_pred, threshold: float) -> dict[str, float | int]:
    each_true = np.asarray(each_true, dtype=float)
    each_pred = np.asarray(each_pred, dtype=float)
    tp_m, fn_m, fp_m, tn_m = classify_stable(each_true, each_pred, threshold)
    tp, fn, fp, tn = map(lambda x: int(np.sum(x)), (tp_m, fn_m, fp_m, tn_m))
    total_pos = tp + fn
    total_neg = tn + fp
    total = len(each_true)
    prevalence = safe_div(total_pos, total_pos + total_neg)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, total_pos)
    f1 = (
        safe_div(2 * precision * recall, precision + recall)
        if np.isfinite(precision) and np.isfinite(recall)
        else float("nan")
    )
    ok = np.isfinite(each_true) & np.isfinite(each_pred)
    err = each_pred[ok] - each_true[ok]
    return {
        "n": int(total),
        "threshold_ev_atom": float(threshold),
        "threshold_mev_atom": float(threshold * 1000.0),
        "prevalence": prevalence,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "Accuracy": safe_div(tp + tn, total),
        "DAF": safe_div(precision, prevalence) if np.isfinite(precision) and np.isfinite(prevalence) else float("nan"),
        "TPR": recall,
        "FPR": safe_div(fp, total_neg),
        "TNR": safe_div(tn, total_neg),
        "FNR": safe_div(fn, total_pos),
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "MAE_ev_atom": float(np.mean(np.abs(err))) if len(err) else float("nan"),
        "RMSE_ev_atom": float(np.sqrt(np.mean(err**2))) if len(err) else float("nan"),
        "R2": r2_score_np(each_true, each_pred),
    }


def add_metric_row(rows: list[dict[str, object]], frame: pd.DataFrame, scope: str, threshold: float, source: Path) -> None:
    row = stable_metrics(
        frame["official_fixed_hull_true_e_above_hull_ev_atom"],
        frame["official_fixed_hull_pred_e_above_hull_ev_atom"],
        threshold,
    )
    row.update(
        {
            "scope": scope,
            "source_rows": str(source),
            "prediction_formula": "e_above_hull_pred = e_above_hull_true + e_form_pred - e_form_dft",
        }
    )
    rows.append(row)


def topk_table(frame: pd.DataFrame, scope: str, unique_chemsys: bool) -> pd.DataFrame:
    pool = frame.sort_values(["official_fixed_hull_pred_e_above_hull_ev_atom", "id"], ascending=[True, True]).copy()
    selection = "lowest_predicted_fixed_hull"
    if unique_chemsys:
        pool = pool.drop_duplicates("chemical_system", keep="first")
        selection += "_unique_chemsys"
    out_rows: list[dict[str, object]] = []
    for threshold in THRESHOLDS:
        pool_stable = (pool["official_fixed_hull_true_e_above_hull_ev_atom"] <= threshold).to_numpy()
        pool_frac = float(np.mean(pool_stable)) if len(pool_stable) else float("nan")
        pool_n_stable = int(np.sum(pool_stable))
        for k in TOP_K:
            if k > len(pool):
                continue
            top = pool.head(k)
            stable = top["official_fixed_hull_true_e_above_hull_ev_atom"] <= threshold
            precision = float(stable.mean()) if len(top) else float("nan")
            n_stable = int(stable.sum())
            out_rows.append(
                {
                    "scope": scope,
                    "selection": selection,
                    "k": int(k),
                    "threshold_ev_atom": float(threshold),
                    "threshold_mev_atom": float(threshold * 1000.0),
                    "pool_n": int(len(pool)),
                    "pool_true_stable_n": int(pool_n_stable),
                    "pool_true_stable_fraction": pool_frac,
                    "topk_true_stable_n": int(n_stable),
                    "topk_precision": precision,
                    "topk_recall_vs_pool": safe_div(n_stable, pool_n_stable),
                    "EF_or_DAF": safe_div(precision, pool_frac) if np.isfinite(pool_frac) else float("nan"),
                    "mean_predicted_ehull_topk_mev_atom": float(
                        top["official_fixed_hull_pred_e_above_hull_ev_atom"].mean() * 1000.0
                    ),
                    "mean_true_ehull_topk_mev_atom": float(
                        top["official_fixed_hull_true_e_above_hull_ev_atom"].mean() * 1000.0
                    ),
                }
            )
    return pd.DataFrame(out_rows)


def fmt(value: object, digits: int = 3) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "nan"
    return "nan" if not np.isfinite(val) else f"{val:.{digits}f}"


def load_official_stable_metrics(wheel: Path):
    if not wheel.exists():
        return None, f"official wheel not found: {wheel}"
    try:
        stub = types.ModuleType("matbench_discovery")
        stub.STABILITY_THRESHOLD = 0
        old_module = sys.modules.get("matbench_discovery")
        sys.modules["matbench_discovery"] = stub
        module = types.ModuleType("matbench_discovery_official_metrics_from_wheel")
        with zipfile.ZipFile(wheel) as zf:
            code = zf.read("matbench_discovery/metrics.py")
        exec(compile(code, str(wheel) + ":matbench_discovery/metrics.py", "exec"), module.__dict__)
        if old_module is not None:
            sys.modules["matbench_discovery"] = old_module
        else:
            del sys.modules["matbench_discovery"]
        return module.stable_metrics, None
    except Exception as exc:  # pragma: no cover - depends on the remote wheel environment
        return None, f"{type(exc).__name__}: {exc}"


def build_official_crosscheck(
    metrics_df: pd.DataFrame,
    scopes: dict[str, pd.DataFrame],
    out_dir: Path,
    wheel: Path,
) -> dict[str, object]:
    official_stable_metrics, error = load_official_stable_metrics(wheel)
    if official_stable_metrics is None:
        return {
            "status": "skipped",
            "wheel": str(wheel),
            "reason": error,
        }

    records: list[dict[str, object]] = []
    metric_keys = [
        "Precision",
        "Recall",
        "F1",
        "Accuracy",
        "DAF",
        "TPR",
        "FPR",
        "TNR",
        "FNR",
        "TP",
        "FP",
        "TN",
        "FN",
        "MAE",
        "RMSE",
        "R2",
    ]
    for scope, frame in scopes.items():
        for threshold in THRESHOLDS:
            official = official_stable_metrics(
                frame["official_fixed_hull_true_e_above_hull_ev_atom"],
                frame["official_fixed_hull_pred_e_above_hull_ev_atom"],
                stability_threshold=threshold,
            )
            script_row = metrics_df[
                (metrics_df["scope"] == scope) & np.isclose(metrics_df["threshold_ev_atom"], threshold)
            ].iloc[0]
            record: dict[str, object] = {"scope": scope, "threshold_ev_atom": threshold}
            for key in metric_keys:
                script_key = {"MAE": "MAE_ev_atom", "RMSE": "RMSE_ev_atom"}.get(key, key)
                official_val = official[key]
                script_val = script_row[script_key]
                record[f"official_{key}"] = official_val
                record[f"script_{key}"] = script_val
                record[f"abs_diff_{key}"] = abs(float(official_val) - float(script_val))
            records.append(record)

    crosscheck = pd.DataFrame(records)
    crosscheck.to_csv(out_dir / "official_matbench_metrics_crosscheck.csv", index=False)
    max_diff = float(crosscheck.filter(like="abs_diff_").to_numpy().max())
    return {
        "status": "passed",
        "wheel": str(wheel),
        "file": "official_matbench_metrics_crosscheck.csv",
        "max_abs_metric_difference": max_diff,
        "method": (
            "Loaded matbench_discovery/metrics.py directly from the official wheel with "
            "a minimal STABILITY_THRESHOLD stub, avoiding optional top-level package dependencies."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Post-process WBM5k predictions with Matbench Discovery fixed-hull metrics.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--official-wheel", type=Path, default=DEFAULT_OFFICIAL_WHEEL)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input, keep_default_na=False)
    numeric_cols = [
        "formation_energy_per_atom_ev_corrected",
        "pred_corrected_per_atom_ev",
        "formal_true_e_hull_ev_atom",
        "formal_pred_e_hull_ev_atom",
        "published_e_hull_per_atom_ev",
        "n_elements",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["official_fixed_hull_true_e_above_hull_ev_atom"] = df["formal_true_e_hull_ev_atom"]
    df["official_fixed_hull_pred_e_above_hull_ev_atom"] = (
        df["formal_true_e_hull_ev_atom"]
        + df["pred_corrected_per_atom_ev"]
        - df["formation_energy_per_atom_ev_corrected"]
    )
    df["official_fixed_hull_pred_minus_existing_formal_pred_abs_ev_atom"] = (
        df["official_fixed_hull_pred_e_above_hull_ev_atom"] - df["formal_pred_e_hull_ev_atom"]
    ).abs()
    df["official_fixed_hull_formation_energy_error_ev_atom"] = (
        df["pred_corrected_per_atom_ev"] - df["formation_energy_per_atom_ev_corrected"]
    )
    df["is_multinary_2plus"] = df["is_unary"].astype(str).str.lower().ne("true") & (df["n_elements"] >= 2)
    df["is_multinary_3plus"] = df["is_unary"].astype(str).str.lower().ne("true") & (df["n_elements"] >= 3)

    row_cols = [
        "id",
        "split",
        "formula",
        "n_elements",
        "is_unary",
        "chemical_system",
        "formation_energy_per_atom_ev_corrected",
        "pred_corrected_per_atom_ev",
        "official_fixed_hull_true_e_above_hull_ev_atom",
        "official_fixed_hull_pred_e_above_hull_ev_atom",
        "formal_pred_e_hull_ev_atom",
        "official_fixed_hull_pred_minus_existing_formal_pred_abs_ev_atom",
        "official_fixed_hull_formation_energy_error_ev_atom",
        "published_e_hull_per_atom_ev",
        "published_stable_le_0p025",
    ]
    df[row_cols].to_csv(out_dir / "matbench_discovery_fixed_hull_row_predictions.csv", index=False)

    scopes = {
        "all_heldout_5000": df,
        "multinary_2plus_heldout": df[df["is_multinary_2plus"]].copy(),
        "multinary_3plus_heldout": df[df["is_multinary_3plus"]].copy(),
    }
    metric_rows: list[dict[str, object]] = []
    for scope, frame in scopes.items():
        for threshold in THRESHOLDS:
            add_metric_row(metric_rows, frame, scope, threshold, args.input)
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(out_dir / "matbench_discovery_fixed_hull_metrics.csv", index=False)
    official_crosscheck = build_official_crosscheck(metrics_df, scopes, out_dir, args.official_wheel)

    topk_frames = []
    for scope, frame in scopes.items():
        if scope == "all_heldout_5000":
            continue
        topk_frames.append(topk_table(frame, scope, unique_chemsys=False))
        topk_frames.append(topk_table(frame, scope, unique_chemsys=True))
    topk_df = pd.concat(topk_frames, ignore_index=True)
    topk_df.to_csv(out_dir / "label_free_topk_enrichment.csv", index=False)

    primary = scopes["multinary_3plus_heldout"].sort_values(
        ["official_fixed_hull_pred_e_above_hull_ev_atom", "id"], ascending=[True, True]
    )
    primary_unique = primary.drop_duplicates("chemical_system", keep="first")
    shortlist_cols = [
        "id",
        "formula",
        "chemical_system",
        "n_elements",
        "formation_energy_per_atom_ev_corrected",
        "pred_corrected_per_atom_ev",
        "official_fixed_hull_true_e_above_hull_ev_atom",
        "official_fixed_hull_pred_e_above_hull_ev_atom",
        "published_e_hull_per_atom_ev",
        "published_stable_le_0p025",
    ]
    for name, pool in [("all_multinary_3plus", primary), ("unique_chemsys_multinary_3plus", primary_unique)]:
        for k in [20, 50, 100]:
            shortlist = pool.head(k)[shortlist_cols].copy()
            shortlist.insert(0, "selection_rank", range(1, len(shortlist) + 1))
            shortlist.insert(0, "shortlist_name", f"label_free_{name}_top{k}")
            shortlist.to_csv(out_dir / f"label_free_{name}_top{k}.csv", index=False)

    max_diff = float(df["official_fixed_hull_pred_minus_existing_formal_pred_abs_ev_atom"].max())
    mean_diff = float(df["official_fixed_hull_pred_minus_existing_formal_pred_abs_ev_atom"].mean())
    summary = {
        "source_predictions": str(args.input),
        "output_dir": str(out_dir),
        "official_formula": "e_above_hull_pred = e_above_hull_true + e_form_pred - e_form_dft",
        "official_sources_checked": [
            "matbench_discovery-1.3.1 wheel: matbench_discovery/metrics.py stable_metrics",
            "fairchem_core materials_discovery_reducer.py lines 333-350",
            "https://raw.githubusercontent.com/janosh/matbench-discovery/main/matbench_discovery/metrics/discovery.py",
        ],
        "row_count": int(len(df)),
        "multinary_2plus_count": int(len(scopes["multinary_2plus_heldout"])),
        "multinary_3plus_count": int(len(scopes["multinary_3plus_heldout"])),
        "max_abs_difference_official_pred_ehull_vs_existing_formal_pred_ev_atom": max_diff,
        "mean_abs_difference_official_pred_ehull_vs_existing_formal_pred_ev_atom": mean_diff,
        "files": [
            "matbench_discovery_fixed_hull_row_predictions.csv",
            "matbench_discovery_fixed_hull_metrics.csv",
            "label_free_topk_enrichment.csv",
            "label_free_all_multinary_3plus_top20.csv",
            "label_free_all_multinary_3plus_top50.csv",
            "label_free_all_multinary_3plus_top100.csv",
            "label_free_unique_chemsys_multinary_3plus_top20.csv",
            "label_free_unique_chemsys_multinary_3plus_top50.csv",
            "label_free_unique_chemsys_multinary_3plus_top100.csv",
            "summary.json",
            "README.md",
        ],
        "official_stable_metrics_crosscheck": official_crosscheck,
    }
    if official_crosscheck.get("file") and official_crosscheck["file"] not in summary["files"]:
        summary["files"].append(official_crosscheck["file"])
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Matbench Discovery fixed-hull WBM5k evaluation",
        "",
        "This is a post-processing-only evaluation; no model training, DFT, or candidate relaxation was run.",
        "",
        f"Source predictions: `{args.input}`.",
        "",
        "Fixed-hull convention:",
        "",
        "`E_hull_pred = E_hull_true + E_form_pred - E_form_DFT`.",
        "",
        f"Max absolute difference between this fixed-hull prediction and existing `formal_pred_e_hull_ev_atom`: `{max_diff:.3e}` eV/atom.",
        "",
        "Official metrics cross-check:",
        "",
        f"`{official_crosscheck['status']}`"
        + (
            f"; max absolute metric difference `{official_crosscheck['max_abs_metric_difference']:.3e}`."
            if official_crosscheck["status"] == "passed"
            else f"; {official_crosscheck.get('reason', 'not available')}."
        ),
        "",
        "## Stability Metrics",
        "",
    ]
    selected_metrics = metrics_df[metrics_df["scope"].isin(["all_heldout_5000", "multinary_3plus_heldout"])]
    for _, row in selected_metrics.iterrows():
        lines.append(
            f"- `{row['scope']}`, threshold `{float(row['threshold_mev_atom']):.0f}` meV: "
            f"P={fmt(row['Precision'])}, R={fmt(row['Recall'])}, F1={fmt(row['F1'])}, "
            f"Acc={fmt(row['Accuracy'])}, DAF={fmt(row['DAF'])}, "
            f"MAE_hull={fmt(float(row['MAE_ev_atom']) * 1000.0)} meV/atom."
        )
    lines.extend(["", "## Label-Free Top-k", ""])
    primary_topk = topk_df[
        (topk_df["scope"] == "multinary_3plus_heldout")
        & (topk_df["selection"] == "lowest_predicted_fixed_hull")
        & (topk_df["threshold_mev_atom"].isin([25.0, 50.0]))
        & (topk_df["k"].isin([20, 50, 100]))
    ]
    for _, row in primary_topk.iterrows():
        lines.append(
            f"- all multinary 3+ top{int(row['k'])}, threshold `{float(row['threshold_mev_atom']):.0f}` meV: "
            f"pool stable fraction={fmt(row['pool_true_stable_fraction'])}, "
            f"top-k precision={fmt(row['topk_precision'])}, EF={fmt(row['EF_or_DAF'])}, "
            f"recall={fmt(row['topk_recall_vs_pool'])}."
        )
    lines.extend(
        [
            "",
            "Boundary: label-free top-k does not filter by `published_stable_le_0p025`; it uses multinary held-out rows and ranks by predicted fixed-hull distance only.",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {out_dir}")
    print(json.dumps(summary, indent=2))
    print("\nmetrics")
    print(metrics_df.to_string(index=False))
    print("\nlabel-free primary top-k")
    print(primary_topk.to_string(index=False))


if __name__ == "__main__":
    main()
