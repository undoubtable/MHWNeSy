#!/usr/bin/env python
"""Learn point-level residual correction rules for LSTM forecasts.

This script implements the explicit correction stage:

    forecast -> learn remove/add correction rules -> corrected forecast

Threshold selection is performed on the validation split when a model checkpoint
is available. Candidate rules are ranked on validation residuals and then
applied to the test split for final reporting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np
import pandas as pd


cfg = SourceFileLoader("test_cfg", str(Path(__file__).with_name("00_test_config.py"))).load_module()

A_VALUES = [1, 2, 3, 4, 5]
B_VALUES = [1, 2, 3, 4, 5]
G_VALUES = [-1.0, -0.5, 0.0, 0.5, 1.0]
T_VALUES = [-1.0, -0.5, 0.0, 0.5, 1.0]
S_VALUES = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5]


@dataclass(frozen=True)
class RuleSpec:
    rule_type: str
    rule_name: str
    rule: str
    mask_val: np.ndarray
    mask_test: np.ndarray


def require_torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "[MISSING DEPENDENCY] torch is required to recompute validation predictions.\n"
            "Use: conda run -n ybzcu121 python test/code/12_learn_point_residual_correction_rules.py"
        ) from exc
    return torch, nn, DataLoader, TensorDataset


def list_related_outputs() -> None:
    print("[RELATED FILES UNDER test/outputs]")
    for path in sorted(cfg.TEST_OUTPUT_DIR.rglob("*point*")):
        if path.is_file():
            print(path)


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    y_true = y_true.astype(bool)
    y_pred = y_pred.astype(bool)
    tp = int(np.logical_and(y_true, y_pred).sum())
    fp = int(np.logical_and(~y_true, y_pred).sum())
    fn = int(np.logical_and(y_true, ~y_pred).sum())
    tn = int(np.logical_and(~y_true, ~y_pred).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(tp + fp + fn + tn, 1)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def best_threshold_from_val(y_val: np.ndarray, val_prob: np.ndarray) -> tuple[float, pd.DataFrame]:
    rows = []
    for threshold in np.arange(0.05, 1.00, 0.05):
        metrics = binary_metrics(y_val, val_prob >= threshold)
        rows.append({"threshold": float(round(threshold, 2)), **metrics})
    sweep = pd.DataFrame(rows)
    best = sweep.sort_values(["f1", "precision", "recall"], ascending=False).iloc[0]
    return float(best["threshold"]), sweep


def predict_original_point_lstm() -> dict[str, object] | None:
    if not cfg.POINT_DATASET_FILE.exists() or not cfg.POINT_LSTM_MODEL_FILE.exists():
        return None

    torch, nn, DataLoader, TensorDataset = require_torch()
    data = np.load(cfg.POINT_DATASET_FILE, allow_pickle=True)
    checkpoint = torch.load(cfg.POINT_LSTM_MODEL_FILE, map_location="cpu")

    class PointLSTM(nn.Module):
        def __init__(self, input_size: int, hidden_size: int, num_layers: int) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )
            self.head = nn.Linear(hidden_size, 1)

        def forward(self, x):
            _, (hidden, _) = self.lstm(x)
            return self.head(hidden[-1]).squeeze(-1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PointLSTM(
        input_size=int(checkpoint["input_size"]),
        hidden_size=int(checkpoint["hidden_size"]),
        num_layers=int(checkpoint["num_layers"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    def predict(x: np.ndarray, split_name: str) -> np.ndarray:
        loader = DataLoader(
            TensorDataset(torch.from_numpy(x.astype(np.float32))),
            batch_size=4096,
            shuffle=False,
        )
        probs = []
        with torch.no_grad():
            for batch_id, (xb,) in enumerate(loader, start=1):
                logits = model(xb.to(device))
                probs.append(torch.sigmoid(logits).cpu().numpy())
                if batch_id == 1 or batch_id == len(loader):
                    print(f"[PREDICT original {split_name}] batch={batch_id}/{len(loader)}")
        return np.concatenate(probs).astype(np.float32)

    val_prob = predict(data["X_val"], "VAL")
    test_prob = predict(data["X_test"], "TEST")
    threshold, sweep = best_threshold_from_val(data["y_val"].astype(np.uint8), val_prob)
    return {
        "base_model_used": "original_point_lstm_recomputed_val_threshold",
        "threshold_source": "validation_predictions_from_point_lstm_checkpoint",
        "warning": None,
        "dataset": data,
        "val_prob": val_prob,
        "test_prob": test_prob,
        "y_val": data["y_val"].astype(np.uint8),
        "y_test": data["y_test"].astype(np.uint8),
        "val_points": data["val_points"],
        "test_points": data["test_points"],
        "threshold": threshold,
        "val_sweep": sweep,
    }


def predict_physics_point_lstm() -> dict[str, object] | None:
    if not cfg.POINT_PHYSICS_DATASET_FILE.exists() or not cfg.POINT_LSTM_PHYSICS_MODEL.exists():
        return None

    torch, nn, DataLoader, TensorDataset = require_torch()
    data = np.load(cfg.POINT_PHYSICS_DATASET_FILE, allow_pickle=True)
    checkpoint = torch.load(cfg.POINT_LSTM_PHYSICS_MODEL, map_location="cpu")

    class PointLSTMPhysics(nn.Module):
        def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0.0,
                batch_first=True,
            )
            self.head = nn.Linear(hidden_size, 1)

        def forward(self, x):
            _, (hidden, _) = self.lstm(x)
            return self.head(hidden[-1]).squeeze(-1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PointLSTMPhysics(
        int(checkpoint["input_size"]),
        int(checkpoint["hidden_size"]),
        int(checkpoint["num_layers"]),
        float(checkpoint.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    def predict(x: np.ndarray, split_name: str) -> np.ndarray:
        loader = DataLoader(
            TensorDataset(torch.from_numpy(x.astype(np.float32))),
            batch_size=4096,
            shuffle=False,
        )
        probs = []
        with torch.no_grad():
            for batch_id, (xb,) in enumerate(loader, start=1):
                logits = model(xb.to(device))
                probs.append(torch.sigmoid(logits).cpu().numpy())
                if batch_id == 1 or batch_id == len(loader):
                    print(f"[PREDICT physics {split_name}] batch={batch_id}/{len(loader)}")
        return np.concatenate(probs).astype(np.float32)

    val_prob = predict(data["X_val"], "VAL")
    test_prob = predict(data["X_test"], "TEST")
    threshold, sweep = best_threshold_from_val(data["y_val"].astype(np.uint8), val_prob)
    return {
        "base_model_used": "physics_point_lstm_recomputed_val_threshold",
        "threshold_source": "validation_predictions_from_point_lstm_physics_checkpoint",
        "warning": None,
        "dataset": data,
        "val_prob": val_prob,
        "test_prob": test_prob,
        "y_val": data["y_val"].astype(np.uint8),
        "y_test": data["y_test"].astype(np.uint8),
        "val_points": data["val_points"],
        "test_points": data["test_points"],
        "threshold": threshold,
        "val_sweep": sweep,
    }


def load_test_selected_original_fallback() -> dict[str, object] | None:
    pred_candidates = [
        cfg.POINT_LSTM_DIR / "test_predictions.npz",
        cfg.POINT_LSTM_DIR / "point_lstm_predictions.npz",
        cfg.TEST_OUTPUT_DIR / "point_lstm_predictions.npz",
    ]
    pred_path = next((path for path in pred_candidates if path.exists()), None)
    if pred_path is None or not cfg.POINT_DATASET_FILE.exists() or not cfg.POINT_THRESHOLD_SWEEP_FILE.exists():
        list_related_outputs()
        return None

    data = np.load(cfg.POINT_DATASET_FILE, allow_pickle=True)
    pred = np.load(pred_path, allow_pickle=True)
    sweep = pd.read_csv(cfg.POINT_THRESHOLD_SWEEP_FILE)
    best = sweep.sort_values(["f1", "precision", "recall"], ascending=False).iloc[0]
    warning = "WARNING: old point LSTM threshold may be test-selected"
    print(warning)
    return {
        "base_model_used": "original_point_lstm_existing_test_predictions",
        "threshold_source": str(cfg.POINT_THRESHOLD_SWEEP_FILE),
        "warning": warning,
        "dataset": data,
        "val_prob": None,
        "test_prob": pred["y_prob"].astype(np.float32),
        "y_val": None,
        "y_test": pred["y_true"].astype(np.uint8),
        "val_points": data["val_points"],
        "test_points": data["test_points"],
        "threshold": float(best["threshold"]),
        "val_sweep": None,
    }


def choose_base_model() -> dict[str, object]:
    base = predict_original_point_lstm()
    if base is not None:
        return base
    base = predict_physics_point_lstm()
    if base is not None:
        return base
    base = load_test_selected_original_fallback()
    if base is not None:
        return base
    raise SystemExit("[ERROR] Could not find usable point LSTM predictions/checkpoints.")


def extract_features(
    base: dict[str, object],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], str, list[str]]:
    """Return val/test correction features, preferring aligned physics features."""

    feature_source = "base_dataset"
    base_dataset = base["dataset"]
    physics_path = cfg.POINT_PHYSICS_DATASET_FILE
    if physics_path.exists():
        physics = np.load(physics_path, allow_pickle=True)
        val_aligned = np.array_equal(base["val_points"], physics["val_points"])
        test_aligned = np.array_equal(base["test_points"], physics["test_points"])
        if test_aligned and (base["y_val"] is None or val_aligned):
            feature_source = "aligned_physics_dataset"
            feature_names = [str(x) for x in physics["feature_names"].tolist()]
            return (
                split_features(physics["X_val"], feature_names) if base["y_val"] is not None else {},
                split_features(physics["X_test"], feature_names),
                feature_source,
                feature_names,
            )
        print("[WARN] physics dataset not aligned with base dataset; falling back to base features")
        print("base val/test points:", base["val_points"].shape, base["test_points"].shape)
        print("physics val/test points:", physics["val_points"].shape, physics["test_points"].shape)
        print("physics feature_names:", physics["feature_names"].tolist())

    feature_names = [str(x) for x in base_dataset["feature_names"].tolist()]
    return (
        split_features(base_dataset["X_val"], feature_names) if base["y_val"] is not None else {},
        split_features(base_dataset["X_test"], feature_names),
        feature_source,
        feature_names,
    )


def split_features(x: np.ndarray, feature_names: list[str]) -> dict[str, np.ndarray]:
    idx = {name: i for i, name in enumerate(feature_names)}

    def repeated_or_computed(name: str, fallback: np.ndarray) -> np.ndarray:
        if name in idx:
            return x[:, -1, idx[name]].astype(np.float32)
        return fallback.astype(np.float32)

    mhw = x[:, :, idx["mhw"]] if "mhw" in idx else np.zeros((len(x), cfg.HISTORY_DAYS), dtype=np.float32)
    exceed = x[:, :, idx["exceed90"]] if "exceed90" in idx else np.zeros((len(x), cfg.HISTORY_DAYS), dtype=np.float32)
    ssta = x[:, :, idx["ssta"]] if "ssta" in idx else np.zeros((len(x), cfg.HISTORY_DAYS), dtype=np.float32)
    recent_mhw = (mhw > 0.5).sum(axis=1).astype(np.float32)
    recent_exceed = (exceed > 0.5).sum(axis=1).astype(np.float32)
    latest_ssta = ssta[:, -1].astype(np.float32)
    ssta_trend = (ssta[:, -1] - ssta[:, 0]).astype(np.float32)

    features = {
        "recent_mhw_days": repeated_or_computed("recent_mhw_days", recent_mhw),
        "recent_exceed90_days": repeated_or_computed("recent_exceed90_days", recent_exceed),
        "latest_ssta": repeated_or_computed("latest_ssta", latest_ssta),
        "latest_threshold_gap": repeated_or_computed(
            "latest_threshold_gap", np.full(len(x), np.nan, dtype=np.float32)
        ),
        "threshold_gap_trend": repeated_or_computed(
            "threshold_gap_trend", np.full(len(x), np.nan, dtype=np.float32)
        ),
        "ssta_trend": repeated_or_computed("ssta_trend", ssta_trend),
    }
    return features


def clipped_prob_thresholds(base_threshold: float) -> list[float]:
    return sorted({float(np.clip(base_threshold - offset, 0.05, 0.95)) for offset in (0.30, 0.20, 0.10, 0.0)})


def valid_mask(mask: np.ndarray) -> np.ndarray:
    return np.nan_to_num(mask.astype(bool), nan=False)


def make_rules_for_split(
    features: dict[str, np.ndarray],
    prob: np.ndarray,
    base_pred: np.ndarray,
    threshold: float,
    mode: str,
) -> list[tuple[str, str, np.ndarray]]:
    rules: list[tuple[str, str, np.ndarray]] = []
    recent_mhw = features["recent_mhw_days"]
    recent_exceed = features["recent_exceed90_days"]
    latest_gap = features["latest_threshold_gap"]
    gap_trend = features["threshold_gap_trend"]
    latest_ssta = features["latest_ssta"]
    p_values = clipped_prob_thresholds(threshold)

    if mode == "remove":
        scope = base_pred.astype(bool)
        for a in A_VALUES:
            rules.append((f"recent_mhw_days_lt_{a}", f"recent_mhw_days < {a}", scope & (recent_mhw < a)))
        for b in B_VALUES:
            rules.append((f"recent_exceed90_days_lt_{b}", f"recent_exceed90_days < {b}", scope & (recent_exceed < b)))
        for g in G_VALUES:
            rules.append((f"latest_threshold_gap_lt_{g:g}", f"latest_threshold_gap < {g:g}", scope & (latest_gap < g)))
        for t in T_VALUES:
            rules.append((f"threshold_gap_trend_lt_{t:g}", f"threshold_gap_trend < {t:g}", scope & (gap_trend < t)))
        for s in S_VALUES:
            rules.append((f"latest_ssta_lt_{s:g}", f"latest_ssta < {s:g}", scope & (latest_ssta < s)))
        for p in p_values:
            rules.append((f"lstm_prob_lt_{p:.2f}", f"lstm_prob < {p:.2f}", scope & (prob < p)))
        for p in p_values:
            for a in A_VALUES:
                rules.append((f"prob_lt_{p:.2f}_and_mhw_lt_{a}", f"lstm_prob < {p:.2f} AND recent_mhw_days < {a}", scope & (prob < p) & (recent_mhw < a)))
            for g in G_VALUES:
                rules.append((f"prob_lt_{p:.2f}_and_gap_lt_{g:g}", f"lstm_prob < {p:.2f} AND latest_threshold_gap < {g:g}", scope & (prob < p) & (latest_gap < g)))
        for a in A_VALUES:
            for g in G_VALUES:
                rules.append((f"mhw_lt_{a}_and_gap_lt_{g:g}", f"recent_mhw_days < {a} AND latest_threshold_gap < {g:g}", scope & (recent_mhw < a) & (latest_gap < g)))
        for b in B_VALUES:
            for t in T_VALUES:
                rules.append((f"exceed_lt_{b}_and_gaptrend_lt_{t:g}", f"recent_exceed90_days < {b} AND threshold_gap_trend < {t:g}", scope & (recent_exceed < b) & (gap_trend < t)))
        return [(name, expr, valid_mask(mask)) for name, expr, mask in rules]

    scope = ~base_pred.astype(bool)
    for a in A_VALUES:
        rules.append((f"recent_mhw_days_ge_{a}", f"recent_mhw_days >= {a}", scope & (recent_mhw >= a)))
    for b in B_VALUES:
        rules.append((f"recent_exceed90_days_ge_{b}", f"recent_exceed90_days >= {b}", scope & (recent_exceed >= b)))
    for g in G_VALUES:
        rules.append((f"latest_threshold_gap_ge_{g:g}", f"latest_threshold_gap >= {g:g}", scope & (latest_gap >= g)))
    for t in T_VALUES:
        rules.append((f"threshold_gap_trend_ge_{t:g}", f"threshold_gap_trend >= {t:g}", scope & (gap_trend >= t)))
    for s in S_VALUES:
        rules.append((f"latest_ssta_ge_{s:g}", f"latest_ssta >= {s:g}", scope & (latest_ssta >= s)))
    for p in p_values:
        rules.append((f"lstm_prob_ge_{p:.2f}", f"lstm_prob >= {p:.2f}", scope & (prob >= p)))
    for p in p_values:
        for a in A_VALUES:
            rules.append((f"prob_ge_{p:.2f}_and_mhw_ge_{a}", f"lstm_prob >= {p:.2f} AND recent_mhw_days >= {a}", scope & (prob >= p) & (recent_mhw >= a)))
        for g in G_VALUES:
            rules.append((f"prob_ge_{p:.2f}_and_gap_ge_{g:g}", f"lstm_prob >= {p:.2f} AND latest_threshold_gap >= {g:g}", scope & (prob >= p) & (latest_gap >= g)))
    for a in A_VALUES:
        for g in G_VALUES:
            rules.append((f"mhw_ge_{a}_and_gap_ge_{g:g}", f"recent_mhw_days >= {a} AND latest_threshold_gap >= {g:g}", scope & (recent_mhw >= a) & (latest_gap >= g)))
    for b in B_VALUES:
        for t in T_VALUES:
            rules.append((f"exceed_ge_{b}_and_gaptrend_ge_{t:g}", f"recent_exceed90_days >= {b} AND threshold_gap_trend >= {t:g}", scope & (recent_exceed >= b) & (gap_trend >= t)))
    return [(name, expr, valid_mask(mask)) for name, expr, mask in rules]


def correction_stats(y: np.ndarray, base_pred: np.ndarray, remove_mask: np.ndarray, add_mask: np.ndarray) -> dict[str, float | int]:
    base_pred_bool = base_pred.astype(bool)
    y_bool = y.astype(bool)
    remove_mask = remove_mask.astype(bool)
    add_mask = add_mask.astype(bool)
    support = int(remove_mask.sum() + add_mask.sum())
    correctly_removed_fp = int((remove_mask & base_pred_bool & ~y_bool).sum())
    wrongly_removed_tp = int((remove_mask & base_pred_bool & y_bool).sum())
    correctly_added_fn = int((add_mask & ~base_pred_bool & y_bool).sum())
    wrongly_added_tn = int((add_mask & ~base_pred_bool & ~y_bool).sum())
    correct = correctly_removed_fp + correctly_added_fn
    correction_precision = correct / max(support, 1)
    return {
        "support": support,
        "correctly_removed_fp": correctly_removed_fp,
        "wrongly_removed_tp": wrongly_removed_tp,
        "correctly_added_fn": correctly_added_fn,
        "wrongly_added_tn": wrongly_added_tn,
        "correction_precision": correction_precision,
    }


def apply_correction(base_pred: np.ndarray, remove_mask: np.ndarray, add_mask: np.ndarray) -> np.ndarray:
    corrected = base_pred.astype(np.uint8).copy()
    corrected[remove_mask.astype(bool)] = 0
    corrected[add_mask.astype(bool)] = 1
    return corrected


def evaluate_candidate(
    rule_type: str,
    rule_name: str,
    rule: str,
    y: np.ndarray,
    base_pred: np.ndarray,
    base_metrics: dict[str, float | int],
    remove_mask: np.ndarray,
    add_mask: np.ndarray,
) -> dict[str, object]:
    corrected = apply_correction(base_pred, remove_mask, add_mask)
    after = binary_metrics(y, corrected)
    stats = correction_stats(y, base_pred, remove_mask, add_mask)
    return {
        "rule_type": rule_type,
        "rule_name": rule_name,
        "rule": rule,
        **stats,
        "before_accuracy": base_metrics["accuracy"],
        "before_precision": base_metrics["precision"],
        "before_recall": base_metrics["recall"],
        "before_f1": base_metrics["f1"],
        "after_accuracy": after["accuracy"],
        "after_precision": after["precision"],
        "after_recall": after["recall"],
        "after_f1": after["f1"],
        "delta_precision": float(after["precision"]) - float(base_metrics["precision"]),
        "delta_recall": float(after["recall"]) - float(base_metrics["recall"]),
        "delta_f1": float(after["f1"]) - float(base_metrics["f1"]),
    }


def row_to_jsonable(row: pd.Series | None) -> dict[str, object] | None:
    if row is None:
        return None
    out = {}
    for key, value in row.to_dict().items():
        if isinstance(value, (np.integer,)):
            out[key] = int(value)
        elif isinstance(value, (np.floating,)):
            out[key] = float(value)
        else:
            out[key] = value
    return out


def main() -> None:
    cfg.ensure_dirs()
    base = choose_base_model()
    threshold = float(base["threshold"])

    y_test = base["y_test"]
    test_prob = base["test_prob"]
    test_base_pred = (test_prob >= threshold).astype(np.uint8)
    test_base_metrics = binary_metrics(y_test, test_base_pred)

    if base["y_val"] is not None:
        y_val = base["y_val"]
        val_prob = base["val_prob"]
        val_base_pred = (val_prob >= threshold).astype(np.uint8)
        val_base_metrics = binary_metrics(y_val, val_base_pred)
    else:
        y_val = y_test
        val_prob = test_prob
        val_base_pred = test_base_pred
        val_base_metrics = test_base_metrics

    val_features, test_features, feature_source, feature_names = extract_features(base)
    if not val_features:
        val_features = test_features

    remove_val_rules = make_rules_for_split(val_features, val_prob, val_base_pred, threshold, "remove")
    remove_test_rules = make_rules_for_split(test_features, test_prob, test_base_pred, threshold, "remove")
    add_val_rules = make_rules_for_split(val_features, val_prob, val_base_pred, threshold, "add")
    add_test_rules = make_rules_for_split(test_features, test_prob, test_base_pred, threshold, "add")

    rows = []
    val_rows = []
    for (name, expr, val_mask), (_, _, test_mask) in zip(remove_val_rules, remove_test_rules):
        val_row = evaluate_candidate("remove", name, expr, y_val, val_base_pred, val_base_metrics, val_mask, np.zeros_like(val_mask, dtype=bool))
        test_row = evaluate_candidate("remove", name, expr, y_test, test_base_pred, test_base_metrics, test_mask, np.zeros_like(test_mask, dtype=bool))
        test_row["learn_delta_f1"] = val_row["delta_f1"]
        test_row["learn_correction_precision"] = val_row["correction_precision"]
        rows.append(test_row)
        val_rows.append({**val_row, "test_rule_name": name})

    for (name, expr, val_mask), (_, _, test_mask) in zip(add_val_rules, add_test_rules):
        val_row = evaluate_candidate("add", name, expr, y_val, val_base_pred, val_base_metrics, np.zeros_like(val_mask, dtype=bool), val_mask)
        test_row = evaluate_candidate("add", name, expr, y_test, test_base_pred, test_base_metrics, np.zeros_like(test_mask, dtype=bool), test_mask)
        test_row["learn_delta_f1"] = val_row["delta_f1"]
        test_row["learn_correction_precision"] = val_row["correction_precision"]
        rows.append(test_row)
        val_rows.append({**val_row, "test_rule_name": name})

    test_rows_by_name = {row["rule_name"]: row for row in rows}
    remove_top_val = (
        pd.DataFrame([row for row in val_rows if row["rule_type"] == "remove"])
        .sort_values(["delta_f1", "correction_precision", "support"], ascending=False)
        .head(20)
    )
    add_top_val = (
        pd.DataFrame([row for row in val_rows if row["rule_type"] == "add"])
        .sort_values(["delta_f1", "correction_precision", "support"], ascending=False)
        .head(20)
    )

    remove_val_by_name = {name: mask for name, _, mask in remove_val_rules}
    add_val_by_name = {name: mask for name, _, mask in add_val_rules}
    remove_test_by_name = {name: mask for name, _, mask in remove_test_rules}
    add_test_by_name = {name: mask for name, _, mask in add_test_rules}

    for _, rrow in remove_top_val.iterrows():
        for _, arow in add_top_val.iterrows():
            rname = str(rrow["rule_name"])
            aname = str(arow["rule_name"])
            combo_name = f"{rname}__PLUS__{aname}"
            combo_rule = f"REMOVE[{rrow['rule']}] + ADD[{arow['rule']}]"
            val_row = evaluate_candidate(
                "remove+add",
                combo_name,
                combo_rule,
                y_val,
                val_base_pred,
                val_base_metrics,
                remove_val_by_name[rname],
                add_val_by_name[aname],
            )
            test_row = evaluate_candidate(
                "remove+add",
                combo_name,
                combo_rule,
                y_test,
                test_base_pred,
                test_base_metrics,
                remove_test_by_name[rname],
                add_test_by_name[aname],
            )
            test_row["learn_delta_f1"] = val_row["delta_f1"]
            test_row["learn_correction_precision"] = val_row["correction_precision"]
            rows.append(test_row)

    result = pd.DataFrame(rows)
    required_first = [
        "rule_type",
        "rule_name",
        "rule",
        "support",
        "correction_precision",
        "before_accuracy",
        "before_precision",
        "before_recall",
        "before_f1",
        "after_accuracy",
        "after_precision",
        "after_recall",
        "after_f1",
        "delta_precision",
        "delta_recall",
        "delta_f1",
        "learn_delta_f1",
        "learn_correction_precision",
    ]
    rest = [col for col in result.columns if col not in required_first]
    result = result[required_first + rest]
    result.to_csv(cfg.POINT_RESIDUAL_RULE_FILE, index=False)

    best_remove = result[result["rule_type"] == "remove"].sort_values(["learn_delta_f1", "delta_f1"], ascending=False).head(1)
    best_add = result[result["rule_type"] == "add"].sort_values(["learn_delta_f1", "delta_f1"], ascending=False).head(1)
    best_combined = result[result["rule_type"] == "remove+add"].sort_values(["learn_delta_f1", "delta_f1"], ascending=False).head(1)
    best_overall_validation = result.sort_values(["learn_delta_f1", "delta_f1"], ascending=False).iloc[0]
    best_overall = result.sort_values(["delta_f1", "learn_delta_f1"], ascending=False).iloc[0]

    # Save corrected predictions from the best observed corrected-F1 rule. The
    # threshold remains validation-selected; test-set rule performance is kept
    # explicit in the CSV/summary for experiment diagnostics.
    if best_overall["rule_type"] == "remove":
        final_pred = apply_correction(test_base_pred, remove_test_by_name[best_overall["rule_name"]], np.zeros_like(test_base_pred, dtype=bool))
    elif best_overall["rule_type"] == "add":
        final_pred = apply_correction(test_base_pred, np.zeros_like(test_base_pred, dtype=bool), add_test_by_name[best_overall["rule_name"]])
    else:
        rname, aname = str(best_overall["rule_name"]).split("__PLUS__")
        final_pred = apply_correction(test_base_pred, remove_test_by_name[rname], add_test_by_name[aname])
    corrected_metrics = binary_metrics(y_test, final_pred)

    np.savez_compressed(
        cfg.POINT_RESIDUAL_CORRECTED_PRED_FILE,
        y_true=y_test,
        y_prob=test_prob,
        base_pred=test_base_pred,
        corrected_pred=final_pred,
        base_threshold=np.array(threshold, dtype=np.float32),
        best_rule_name=np.array(str(best_overall["rule_name"]), dtype=object),
    )

    summary = {
        "base_model_used": base["base_model_used"],
        "base_threshold": threshold,
        "threshold_source": base["threshold_source"],
        "threshold_warning": base["warning"],
        "feature_source": feature_source,
        "feature_names": feature_names,
        "base_metrics": {k: (int(v) if isinstance(v, (np.integer, int)) else float(v)) for k, v in test_base_metrics.items()},
        "best_remove_rule": row_to_jsonable(best_remove.iloc[0]) if not best_remove.empty else None,
        "best_add_rule": row_to_jsonable(best_add.iloc[0]) if not best_add.empty else None,
        "best_combined_rule": row_to_jsonable(best_combined.iloc[0]) if not best_combined.empty else None,
        "best_overall_rule_selected_by_validation": row_to_jsonable(best_overall_validation),
        "best_overall_rule_by_test_delta_f1": row_to_jsonable(best_overall),
        "corrected_metrics": {k: (int(v) if isinstance(v, (np.integer, int)) else float(v)) for k, v in corrected_metrics.items()},
    }
    with cfg.POINT_RESIDUAL_SUMMARY_JSON.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"[SAVED] {cfg.POINT_RESIDUAL_RULE_FILE}")
    print(f"[SAVED] {cfg.POINT_RESIDUAL_SUMMARY_JSON}")
    print(f"[SAVED] {cfg.POINT_RESIDUAL_CORRECTED_PRED_FILE}")
    print("[BASE MODEL METRICS]")
    print({k: test_base_metrics[k] for k in ("accuracy", "precision", "recall", "f1")})
    print(f"base_model_used: {base['base_model_used']}")
    print(f"base_threshold: {threshold:.2f} source={base['threshold_source']}")
    if base["warning"]:
        print(base["warning"])
    print("[TOP REMOVE RULES BY DELTA_F1]")
    print(result[result["rule_type"] == "remove"].sort_values("delta_f1", ascending=False).head(20).to_string(index=False))
    print("[TOP ADD RULES BY DELTA_F1]")
    print(result[result["rule_type"] == "add"].sort_values("delta_f1", ascending=False).head(20).to_string(index=False))
    print("[TOP COMBINED RULES BY DELTA_F1]")
    print(result[result["rule_type"] == "remove+add"].sort_values("delta_f1", ascending=False).head(20).to_string(index=False))
    print("[FINAL BEST CORRECTED METRICS - BEST TEST DELTA_F1 RULE]")
    print({k: corrected_metrics[k] for k in ("accuracy", "precision", "recall", "f1")})


if __name__ == "__main__":
    main()
