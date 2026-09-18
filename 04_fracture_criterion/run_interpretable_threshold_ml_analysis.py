# -*- coding: utf-8 -*-
"""
run_interpretable_threshold_ml_analysis.py

功能：
1. 读取新版静态-动态局部化提取结果：
   - all_cases_summary.csv
   - key_metrics_summary.csv
   - key_frame_audit.csv（可选）
2. 合并静态力学指标和断裂形貌标签；
3. 构建一行一个工况的特征表；
4. 对候选物理量进行：
   - 单指标阈值扫描
   - 单层决策树/decision stump 等价筛选
   - ROC-AUC
   - 留一法交叉验证 LOOCV
   - 可选 Logistic regression 概率曲线
   - 可选 Bootstrap 阈值置信区间
5. 输出 Excel/CSV/论文用图片。

说明：
- 本脚本不是黑箱机器学习主模型，而是“机器学习辅助的可解释阈值识别”。
- 默认主候选指标包括：
  delta_gamma_excess_pre_drop,
  delta_E_GB_pre_drop,
  delta_gamma_excess_static_prefracture,
  delta_E_GB_static_prefracture,
  I_static_toughness 等。
- 所有图片标题/子标题放在图下方，满足你的绘图格式要求。

运行前需要确认：
- key_metrics_summary.csv 和 all_cases_summary.csv 已经生成并放在 ./data；
- IG_label 标签文件存在，或者 threshold_labeled_features.xlsx 中有标签；
- 如需合并静态力学退化指标，将 static_mechanical_features_for_coupling.xlsx 放在 ./data。
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False


# =============================================================================
# 1. 路径设置
# =============================================================================

# Repository-relative input/output paths.
# Repository-relative input and output paths are configured below.
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR / "data"

ALL_CASES_CSV = BASE_DIR / "all_cases_summary.csv"
KEY_METRICS_CSV = BASE_DIR / "key_metrics_summary.csv"
AUDIT_CSV = BASE_DIR / "key_frame_audit.csv"

# Static mechanical indicators are optional. If present, they are merged
# with the localization features (e.g., I_static_* indicators).
STATIC_MECH_XLSX = BASE_DIR / "static_mechanical_features_for_coupling.xlsx"
STATIC_MECH_SHEET = "relative_to_0H"

# Label-file priority:
# 1. Manually curated fracture_mode_labels_template.xlsx/csv
# 2. Legacy threshold_labeled_features.xlsx containing IG_label
LABEL_CANDIDATES = [
    BASE_DIR / "fracture_mode_labels_template.xlsx",
    BASE_DIR / "fracture_mode_labels_template.csv",
    BASE_DIR / "threshold_labeled_features.xlsx",
]

OUT_DIR = SCRIPT_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

OUT_FEATURE_CSV = OUT_DIR / "ml_feature_table.csv"
OUT_FEATURE_XLSX = OUT_DIR / "ml_feature_table.xlsx"
OUT_ANALYSIS_XLSX = OUT_DIR / "interpretable_threshold_ml_results.xlsx"


# =============================================================================
# 2. 分析设置
# =============================================================================

SIGMA_ORDER = ["sigma3", "sigma5", "sigma11", "sigma17"]
COV_ORDER = ["0H", "1H", "005cov", "010cov", "015cov", "020cov", "025cov", "050cov", "100cov"]

# 是否把 0H 纳入阈值学习。
# 如果 IG_label 定义为“断裂形貌标签”，可保留 True；
# 如果 IG_label 定义为“氢诱导沿晶开裂标签”，建议改 False。
INCLUDE_0H_IN_ML = True

# 是否把 1H 纳入阈值学习。
INCLUDE_1H_IN_ML = True

# 是否输出 Bootstrap 阈值置信区间。样本少，建议保留作为稳健性辅助。
DO_BOOTSTRAP = True
N_BOOTSTRAP = 1000
RANDOM_SEED = 20260603

# Logistic regression 只作为辅助图，不作为主判据。
DO_LOGISTIC = True

# 候选特征，脚本会自动跳过不存在或有效样本不足的列。
CANDIDATE_FEATURES = [
    # 静态力学退化
    "I_static_strength",
    "I_static_deformability",
    "I_static_toughness",
    "I_static_softening",
    "peak_stress_degradation_pct",
    "peak_strain_advance_pct",
    "toughness_degradation_pct",

    # 静态结构/局部化
    "delta_E_GB_static_global_peak",
    "delta_E_GB_static_prefracture",
    "delta_E_GB_static_prefracture_window",
    "delta_gamma_excess_static_global_peak",
    "delta_gamma_excess_static_prefracture",
    "delta_gamma_excess_static_prefracture_window",

    # 动态结构/局部化
    "delta_E_GB_global_peak",
    "delta_E_GB_pre_drop",
    "delta_E_GB_pre_drop_window",
    "delta_gamma_excess_global_peak",
    "delta_gamma_excess_pre_drop",
    "delta_gamma_excess_pre_drop_window",
    "max_delta_gamma_excess",
    "max_delta_E_GB",
    "area_delta_gamma_excess",
    "area_delta_gamma_excess_pos",
    "max_slope_delta_gamma_excess",
    "max_slope_delta_E_GB",
]

# 主要图推荐使用这个特征；如果它不存在，则自动使用 F1 最优特征。
PREFERRED_PRIMARY_FEATURE = "delta_gamma_excess_pre_drop"


# =============================================================================
# 3. 基础工具函数
# =============================================================================

def normalize_sigma(x) -> str:
    return str(x).strip().lower()


def normalize_cov(x) -> str:
    s = str(x).strip()
    sl = s.lower()
    if sl == "0h":
        return "0H"
    if sl == "1h":
        return "1H"
    return sl if sl.endswith("cov") else s


def sigma_rank(s: str) -> int:
    s = normalize_sigma(s)
    return SIGMA_ORDER.index(s) if s in SIGMA_ORDER else 999


def cov_rank(c: str) -> int:
    c = normalize_cov(c)
    return COV_ORDER.index(c) if c in COV_ORDER else 999


def to_numeric_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def safe_div(a: float, b: float) -> float:
    if b == 0 or not np.isfinite(b):
        return math.nan
    return a / b


def f1_score_manual(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    n = len(y_true)
    acc = safe_div(tp + tn, n)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    f1 = safe_div(2 * precision * recall, precision + recall)
    if not np.isfinite(precision):
        precision = 0.0
    if not np.isfinite(recall):
        recall = 0.0
    if not np.isfinite(specificity):
        specificity = 0.0
    if not np.isfinite(f1):
        f1 = 0.0
    return {
        "n": n,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
    }


def auc_rank(y_true, score) -> float:
    """无 sklearn 时使用的 Mann-Whitney AUC。"""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(score, dtype=float)
    mask = np.isfinite(s)
    y = y[mask]
    s = s[mask]
    if len(np.unique(y)) < 2:
        return math.nan
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return math.nan
    count = 0.0
    total = len(pos) * len(neg)
    for p in pos:
        count += np.sum(p > neg)
        count += 0.5 * np.sum(p == neg)
    return count / total


def compute_auc(y_true, x, direction="greater") -> float:
    score = np.asarray(x, dtype=float)
    if direction == "less":
        score = -score
    if SKLEARN_OK:
        try:
            return float(roc_auc_score(y_true, score))
        except Exception:
            return math.nan
    return auc_rank(y_true, score)


def candidate_thresholds(x: np.ndarray) -> np.ndarray:
    vals = np.sort(np.unique(x[np.isfinite(x)]))
    if len(vals) < 2:
        return np.array([], dtype=float)
    mids = (vals[:-1] + vals[1:]) / 2.0
    # 也加入唯一值本身，避免极端小样本时错过边界。
    thrs = np.unique(np.concatenate([mids, vals]))
    return thrs[np.isfinite(thrs)]


def scan_one_feature(df: pd.DataFrame, feature: str, label_col="IG_label") -> pd.DataFrame:
    if feature not in df.columns or label_col not in df.columns:
        return pd.DataFrame()
    tmp = df[["sigma", "cov", feature, label_col]].copy()
    tmp[feature] = to_numeric_series(tmp[feature])
    tmp[label_col] = to_numeric_series(tmp[label_col])
    tmp = tmp[np.isfinite(tmp[feature]) & tmp[label_col].isin([0, 1])]
    if len(tmp) < 4 or tmp[label_col].nunique() < 2:
        return pd.DataFrame()

    x = tmp[feature].to_numpy(dtype=float)
    y = tmp[label_col].to_numpy(dtype=int)
    thrs = candidate_thresholds(x)
    rows = []
    for direction in ["greater", "less"]:
        for c in thrs:
            pred = (x > c).astype(int) if direction == "greater" else (x < c).astype(int)
            met = f1_score_manual(y, pred)
            auc_val = compute_auc(y, x, direction=direction)
            rows.append({
                "feature": feature,
                "threshold": float(c),
                "direction": direction,
                "auc": auc_val,
                **met,
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    # 排序准则：F1 优先，其次 accuracy、recall、specificity、AUC。
    out = out.sort_values(
        by=["f1", "accuracy", "recall", "specificity", "auc"],
        ascending=[False, False, False, False, False]
    ).reset_index(drop=True)
    return out


def find_best_threshold(df: pd.DataFrame, features: List[str], label_col="IG_label") -> Tuple[pd.DataFrame, pd.DataFrame]:
    all_rows = []
    best_rows = []
    for feat in features:
        res = scan_one_feature(df, feat, label_col=label_col)
        if res.empty:
            continue
        all_rows.append(res)
        best_rows.append(res.iloc[[0]])
    all_scan = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    best = pd.concat(best_rows, ignore_index=True) if best_rows else pd.DataFrame()
    if not best.empty:
        best = best.sort_values(
            by=["f1", "accuracy", "recall", "specificity", "auc"],
            ascending=[False, False, False, False, False]
        ).reset_index(drop=True)
    return all_scan, best


def apply_threshold(x: pd.Series, threshold: float, direction: str) -> pd.Series:
    xx = to_numeric_series(x)
    if direction == "greater":
        return (xx > threshold).astype(float)
    return (xx < threshold).astype(float)


def bottom_title(fig, text, fontsize=13, y=0.01):
    fig.text(0.5, y, text, ha="center", va="bottom", fontsize=fontsize, fontweight="bold")


def save_fig(fig, out_base: Path):
    fig.savefig(str(out_base) + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(str(out_base) + ".pdf", bbox_inches="tight")
    fig.savefig(str(out_base) + ".svg", bbox_inches="tight")
    plt.close(fig)


def pretty_feature(name: str) -> str:
    m = {
        "delta_gamma_excess_pre_drop": r"$\Delta\gamma_{excess}^{D}$",
        "delta_E_GB_pre_drop": r"$\Delta E_{GB}^{D}$",
        "I_static_toughness": r"$\eta_U^S$ (%)",
        "I_static_strength": r"$\eta_{\sigma}^S$ (%)",
        "I_static_deformability": r"$\eta_{\epsilon}^S$ (%)",
        "delta_gamma_excess_static_prefracture": r"$\Delta\gamma_{excess}^{S,pre-f}$",
        "delta_E_GB_static_prefracture": r"$\Delta E_{GB}^{S,pre-f}$",
        "max_delta_gamma_excess": r"max$(\Delta\gamma_{excess})$",
        "area_delta_gamma_excess_pos": r"Area$_+(\Delta\gamma_{excess})$",
        "max_slope_delta_gamma_excess": r"max$(d\Delta\gamma_{excess}/d\epsilon)$",
    }
    return m.get(name, name.replace("_", "\\_"))


def pretty_sigma(s: str) -> str:
    return {"sigma3": "Σ3", "sigma5": "Σ5", "sigma11": "Σ11", "sigma17": "Σ17"}.get(normalize_sigma(s), str(s))


def pretty_cov(c: str) -> str:
    return {
        "0H": "0H", "1H": "1H", "005cov": "5%", "010cov": "10%", "015cov": "15%",
        "020cov": "20%", "025cov": "25%", "050cov": "50%", "100cov": "100%",
    }.get(normalize_cov(c), str(c))


# =============================================================================
# 4. 读取和整理输入数据
# =============================================================================

def read_table_any(path: Path, sheet_name: Optional[str] = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    suffix = path.suffix.lower()
    if suffix in [".xlsx", ".xls"]:
        try:
            return pd.read_excel(path, sheet_name=sheet_name if sheet_name is not None else 0)
        except Exception:
            return pd.read_excel(path, sheet_name=0)
    return pd.read_csv(path, encoding="utf-8-sig")


def normalize_case_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    if "sigma" in df.columns:
        df["sigma"] = df["sigma"].map(normalize_sigma)
    if "cov" in df.columns:
        df["cov"] = df["cov"].map(normalize_cov)
    return df


def load_labels() -> pd.DataFrame:
    for p in LABEL_CANDIDATES:
        if not p.exists():
            continue
        df = read_table_any(p, sheet_name="labeled_features" if "threshold_labeled" in p.name else None)
        if df.empty:
            continue
        df = normalize_case_columns(df)
        if "sigma" not in df.columns or "cov" not in df.columns:
            continue
        keep = ["sigma", "cov"]
        for col in ["IG_label", "fracture_mode", "notes", "label_source"]:
            if col in df.columns:
                keep.append(col)
        if "IG_label" in keep:
            out = df[keep].drop_duplicates(subset=["sigma", "cov"], keep="last")
            out["label_file_used"] = str(p)
            return out
    return pd.DataFrame(columns=["sigma", "cov", "IG_label", "fracture_mode", "notes", "label_file_used"])


def build_key_feature_table(key_df: pd.DataFrame) -> pd.DataFrame:
    key_df = normalize_case_columns(key_df)
    if "mode" not in key_df.columns:
        # 兼容老动态结果：没有 mode 就默认 dynamic。
        key_df["mode"] = "dynamic"

    key_df["mode"] = key_df["mode"].astype(str).str.strip().str.lower()
    key_df["key_tag"] = key_df["key_tag"].astype(str).str.strip()

    metric_cols = [
        "f_nonBCC_GB", "f_nonBCC_grain", "delta_f_nonBCC_GB", "delta_f_nonBCC_grain",
        "L_GB", "E_GB", "delta_E_GB",
        "gamma_GB_mean", "gamma_grain_mean", "delta_gamma_GB", "delta_gamma_grain",
        "L_gamma", "gamma_excess", "delta_gamma_excess",
        "N_Fe_GB", "N_Fe_grain",
        "strain_from_csv", "stress_from_csv", "stress_effective_for_detection",
    ]
    metric_cols = [c for c in metric_cols if c in key_df.columns]
    for c in metric_cols:
        key_df[c] = to_numeric_series(key_df[c])

    rows = []
    for (sigma, cov), g in key_df.groupby(["sigma", "cov"], dropna=False):
        row = {"sigma": sigma, "cov": cov}
        for _, r in g.iterrows():
            mode = str(r.get("mode", "dynamic")).lower()
            key = str(r.get("key_tag", ""))
            if mode == "dynamic":
                if key == "global_peak":
                    suffix = "global_peak"
                elif key == "pre_drop":
                    suffix = "pre_drop"
                elif key == "pre_drop_window_mean":
                    suffix = "pre_drop_window"
                else:
                    continue
                for m in metric_cols:
                    row[f"{m}_{suffix}"] = r.get(m, math.nan)
            elif mode == "static":
                if key == "global_peak":
                    suffix = "static_global_peak"
                elif key == "pre_fracture":
                    suffix = "static_prefracture"
                elif key == "pre_fracture_window_mean":
                    suffix = "static_prefracture_window"
                else:
                    continue
                for m in metric_cols:
                    row[f"{m}_{suffix}"] = r.get(m, math.nan)
        rows.append(row)
    return pd.DataFrame(rows)


def integrate_curve(x: np.ndarray, y: np.ndarray, positive_only=False) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return math.nan
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    if positive_only:
        y = np.maximum(y, 0)
    try:
        return float(np.trapz(y, x))
    except Exception:
        return math.nan


def max_slope(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 3:
        return math.nan
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    dx = np.diff(x)
    dy = np.diff(y)
    valid = np.isfinite(dx) & np.isfinite(dy) & (np.abs(dx) > 1e-12)
    if not np.any(valid):
        return math.nan
    return float(np.nanmax(dy[valid] / dx[valid]))


def build_full_curve_feature_table(all_df: pd.DataFrame) -> pd.DataFrame:
    all_df = normalize_case_columns(all_df)
    if "mode" not in all_df.columns:
        all_df["mode"] = "dynamic"
    all_df["mode"] = all_df["mode"].astype(str).str.lower()

    for c in ["strain_from_csv", "frame", "use_for_quant", "delta_gamma_excess", "delta_E_GB"]:
        if c in all_df.columns:
            all_df[c] = to_numeric_series(all_df[c])

    rows = []
    for (mode, sigma, cov), g in all_df.groupby(["mode", "sigma", "cov"], dropna=False):
        row = {"sigma": sigma, "cov": cov}
        x = g["strain_from_csv"].to_numpy(dtype=float) if "strain_from_csv" in g.columns else g["frame"].to_numpy(dtype=float)
        if np.all(~np.isfinite(x)) and "frame" in g.columns:
            x = g["frame"].to_numpy(dtype=float)
        quant = g["use_for_quant"].to_numpy(dtype=float) == 1 if "use_for_quant" in g.columns else np.ones(len(g), dtype=bool)

        prefix = "" if mode == "dynamic" else "static_"

        for base in ["delta_gamma_excess", "delta_E_GB"]:
            if base not in g.columns:
                continue
            y_all = g[base].to_numpy(dtype=float)
            y_quant = y_all.copy()
            y_quant[~quant] = math.nan
            row[f"max_{prefix}{base}"] = float(np.nanmax(y_quant)) if np.any(np.isfinite(y_quant)) else math.nan
            row[f"min_{prefix}{base}"] = float(np.nanmin(y_quant)) if np.any(np.isfinite(y_quant)) else math.nan
            row[f"area_{prefix}{base}"] = integrate_curve(x, y_quant, positive_only=False)
            row[f"area_{prefix}{base}_pos"] = integrate_curve(x, y_quant, positive_only=True)
            row[f"max_slope_{prefix}{base}"] = max_slope(x, y_quant)
        rows.append(row)
    out = pd.DataFrame(rows)
    # dynamic/static 分两行时要合并到同一行。
    if out.empty:
        return out
    merged = None
    for _, g in out.groupby(["sigma", "cov"], dropna=False):
        r = {"sigma": g["sigma"].iloc[0], "cov": g["cov"].iloc[0]}
        for col in g.columns:
            if col in ["sigma", "cov"]:
                continue
            vals = g[col].dropna()
            if len(vals):
                r[col] = vals.iloc[0]
        merged = pd.concat([merged, pd.DataFrame([r])], ignore_index=True) if merged is not None else pd.DataFrame([r])
    return merged


def load_static_mechanical() -> pd.DataFrame:
    if not STATIC_MECH_XLSX.exists():
        return pd.DataFrame()
    try:
        df = pd.read_excel(STATIC_MECH_XLSX, sheet_name=STATIC_MECH_SHEET)
    except Exception:
        df = pd.read_excel(STATIC_MECH_XLSX, sheet_name=0)
    df = normalize_case_columns(df)
    return df


def merge_tables(tables: List[pd.DataFrame]) -> pd.DataFrame:
    merged = None
    for t in tables:
        if t is None or t.empty:
            continue
        t = normalize_case_columns(t)
        if "sigma" not in t.columns or "cov" not in t.columns:
            continue
        if merged is None:
            merged = t.copy()
        else:
            merged = pd.merge(merged, t, on=["sigma", "cov"], how="outer", suffixes=("", "_dup"))
            dup_cols = [c for c in merged.columns if c.endswith("_dup")]
            for dc in dup_cols:
                base = dc[:-4]
                if base in merged.columns:
                    merged[base] = merged[base].combine_first(merged[dc])
                else:
                    merged[base] = merged[dc]
            merged = merged.drop(columns=dup_cols)
    if merged is None:
        return pd.DataFrame(columns=["sigma", "cov"])
    merged["sigma_rank"] = merged["sigma"].map(sigma_rank)
    merged["cov_rank"] = merged["cov"].map(cov_rank)
    merged = merged.sort_values(["sigma_rank", "cov_rank"]).drop(columns=["sigma_rank", "cov_rank"]).reset_index(drop=True)
    return merged


# =============================================================================
# 5. LOOCV / Bootstrap / 绘图
# =============================================================================

def make_ml_dataset(feature_table: pd.DataFrame) -> pd.DataFrame:
    df = feature_table.copy()
    if "IG_label" not in df.columns:
        raise ValueError("没有 IG_label。请先准备 fracture_mode_labels_template.xlsx/csv 或 threshold_labeled_features.xlsx。")
    df["IG_label"] = to_numeric_series(df["IG_label"])
    df = df[df["IG_label"].isin([0, 1])].copy()
    if not INCLUDE_0H_IN_ML:
        df = df[df["cov"] != "0H"].copy()
    if not INCLUDE_1H_IN_ML:
        df = df[df["cov"] != "1H"].copy()
    return df.reset_index(drop=True)


def loocv_threshold(df: pd.DataFrame, features: List[str], fixed_feature: Optional[str] = None) -> pd.DataFrame:
    rows = []
    for i in range(len(df)):
        train = df.drop(index=i).reset_index(drop=True)
        test = df.iloc[[i]].copy()
        if fixed_feature is not None:
            feats = [fixed_feature]
        else:
            feats = features
        _, best = find_best_threshold(train, feats)
        if best.empty:
            continue
        b = best.iloc[0]
        feat = b["feature"]
        thr = float(b["threshold"])
        direction = str(b["direction"])
        x_val = pd.to_numeric(test[feat], errors="coerce").iloc[0] if feat in test.columns else math.nan
        if not np.isfinite(x_val):
            pred = math.nan
        else:
            pred = int(x_val > thr) if direction == "greater" else int(x_val < thr)
        rows.append({
            "heldout_index": int(i),
            "sigma": test["sigma"].iloc[0],
            "cov": test["cov"].iloc[0],
            "true_label": int(test["IG_label"].iloc[0]),
            "selected_feature": feat,
            "threshold": thr,
            "direction": direction,
            "x_value": x_val,
            "pred_label": pred,
            "correct": int(pred == int(test["IG_label"].iloc[0])) if np.isfinite(pred) else math.nan,
        })
    return pd.DataFrame(rows)


def summarize_loocv(pred_df: pd.DataFrame) -> Dict[str, float]:
    tmp = pred_df[np.isfinite(pd.to_numeric(pred_df["pred_label"], errors="coerce"))].copy()
    if tmp.empty:
        return {}
    y = tmp["true_label"].astype(int).to_numpy()
    p = tmp["pred_label"].astype(int).to_numpy()
    return f1_score_manual(y, p)


def bootstrap_best_threshold(df: pd.DataFrame, feature: str, n_boot=1000) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    valid = df[["sigma", "cov", feature, "IG_label"]].copy()
    valid[feature] = to_numeric_series(valid[feature])
    valid["IG_label"] = to_numeric_series(valid["IG_label"])
    valid = valid[np.isfinite(valid[feature]) & valid["IG_label"].isin([0, 1])].reset_index(drop=True)
    n = len(valid)
    if n < 4:
        return pd.DataFrame()
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = valid.iloc[idx].reset_index(drop=True)
        if sample["IG_label"].nunique() < 2:
            continue
        _, best = find_best_threshold(sample, [feature])
        if best.empty:
            continue
        r = best.iloc[0].to_dict()
        r["bootstrap_id"] = b
        rows.append(r)
    return pd.DataFrame(rows)


def plot_feature_rank(best_df: pd.DataFrame):
    if best_df.empty:
        return
    top = best_df.head(12).copy()
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    yy = np.arange(len(top))[::-1]
    ax.barh(yy, top["f1"].to_numpy())
    ax.set_yticks(yy)
    ax.set_yticklabels([str(f) for f in top["feature"]])
    ax.set_xlabel("Best F1 score")
    ax.set_xlim(0, 1.05)
    for y, f1, acc in zip(yy, top["f1"], top["accuracy"]):
        ax.text(float(f1) + 0.01, y, f"F1={f1:.2f}, Acc={acc:.2f}", va="center", fontsize=8)
    ax.tick_params(direction="in")
    ax.grid(False)
    bottom_title(fig, "(a) Candidate physical indicators ranked by threshold-classification F1", y=0.005)
    fig.subplots_adjust(left=0.36, bottom=0.16)
    save_fig(fig, FIG_DIR / "fig_feature_rank_f1")


def plot_best_feature_threshold(df: pd.DataFrame, best_row: pd.Series):
    feat = str(best_row["feature"])
    thr = float(best_row["threshold"])
    direction = str(best_row["direction"])
    if feat not in df.columns:
        return
    tmp = df[["sigma", "cov", "IG_label", feat]].copy()
    tmp[feat] = to_numeric_series(tmp[feat])
    tmp = tmp[np.isfinite(tmp[feat]) & tmp["IG_label"].isin([0, 1])]
    if tmp.empty:
        return

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    rng = np.random.default_rng(RANDOM_SEED)
    for lab, marker, face in [(0, "o", "none"), (1, "o", "tab:red")]:
        g = tmp[tmp["IG_label"] == lab]
        if g.empty:
            continue
        y_jit = lab + rng.normal(0, 0.025, size=len(g))
        ax.scatter(g[feat], y_jit, marker=marker, s=70,
                   facecolors=face, edgecolors="k", linewidths=1.2, label=f"IG={lab}")
        for _, r in g.iterrows():
            if r["cov"] in ["0H", "025cov", "050cov", "100cov"]:
                ax.text(r[feat], lab + 0.06, f"{pretty_sigma(r['sigma'])}-{pretty_cov(r['cov'])}", fontsize=7, rotation=20)
    ax.axvline(thr, ls="--", lw=1.4, color="0.25", label=f"threshold={thr:.4g}, {direction}")
    ax.set_xlabel(pretty_feature(feat))
    ax.set_ylabel("Fracture label")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["IG=0", "IG=1"])
    ax.legend(frameon=False, loc="best")
    ax.tick_params(direction="in")
    ax.grid(False)
    bottom_title(fig, "(b) Best interpretable threshold for grain-boundary localized cracking", y=0.005)
    fig.subplots_adjust(bottom=0.18)
    save_fig(fig, FIG_DIR / "fig_best_feature_threshold")


def plot_roc(df: pd.DataFrame, best_row: pd.Series):
    feat = str(best_row["feature"])
    direction = str(best_row["direction"])
    if feat not in df.columns:
        return
    tmp = df[[feat, "IG_label"]].copy()
    tmp[feat] = to_numeric_series(tmp[feat])
    tmp["IG_label"] = to_numeric_series(tmp["IG_label"])
    tmp = tmp[np.isfinite(tmp[feat]) & tmp["IG_label"].isin([0, 1])]
    if tmp["IG_label"].nunique() < 2:
        return
    y = tmp["IG_label"].astype(int).to_numpy()
    score = tmp[feat].to_numpy(dtype=float)
    if direction == "less":
        score = -score
    # 手写 ROC
    order_thr = np.sort(np.unique(score))[::-1]
    thresholds = np.r_[np.inf, order_thr, -np.inf]
    tpr, fpr = [], []
    for t in thresholds:
        pred = (score >= t).astype(int)
        met = f1_score_manual(y, pred)
        tpr.append(met["recall"])
        fpr.append(1.0 - met["specificity"])
    auc = compute_auc(y, tmp[feat].to_numpy(dtype=float), direction=direction)
    fig, ax = plt.subplots(figsize=(5.3, 5.0))
    ax.plot(fpr, tpr, lw=1.8, label=f"AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], ls="--", color="0.5", lw=1.0)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, loc="lower right")
    ax.tick_params(direction="in")
    ax.grid(False)
    bottom_title(fig, "(c) ROC curve of the best physical indicator", y=0.005)
    fig.subplots_adjust(bottom=0.16)
    save_fig(fig, FIG_DIR / "fig_roc_best_feature")


def plot_logistic(df: pd.DataFrame, feature: str):
    if not SKLEARN_OK or not DO_LOGISTIC or feature not in df.columns:
        return pd.DataFrame()
    tmp = df[[feature, "IG_label"]].copy()
    tmp[feature] = to_numeric_series(tmp[feature])
    tmp["IG_label"] = to_numeric_series(tmp["IG_label"])
    tmp = tmp[np.isfinite(tmp[feature]) & tmp["IG_label"].isin([0, 1])]
    if len(tmp) < 6 or tmp["IG_label"].nunique() < 2:
        return pd.DataFrame()
    X = tmp[[feature]].to_numpy(dtype=float)
    y = tmp["IG_label"].astype(int).to_numpy()
    model = make_pipeline(StandardScaler(), LogisticRegression(solver="lbfgs"))
    model.fit(X, y)
    xx = np.linspace(np.nanmin(X), np.nanmax(X), 300).reshape(-1, 1)
    prob = model.predict_proba(xx)[:, 1]
    # 找 P=0.5 对应阈值
    idx = int(np.argmin(np.abs(prob - 0.5)))
    p05 = float(xx[idx, 0])

    fig, ax = plt.subplots(figsize=(7.0, 4.7))
    ax.plot(xx[:, 0], prob, lw=1.8, label="Logistic probability")
    ax.scatter(tmp[feature], y, s=60, facecolors="none", edgecolors="k", label="Samples")
    ax.axhline(0.5, ls="--", lw=1.0, color="0.5")
    ax.axvline(p05, ls="--", lw=1.0, color="tab:red", label=f"P=0.5 at {p05:.4g}")
    ax.set_xlabel(pretty_feature(feature))
    ax.set_ylabel("P(IG=1)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=False, loc="best")
    ax.tick_params(direction="in")
    ax.grid(False)
    bottom_title(fig, "(d) Logistic-regression auxiliary probability curve", y=0.005)
    fig.subplots_adjust(bottom=0.18)
    save_fig(fig, FIG_DIR / "fig_logistic_probability")

    return pd.DataFrame([{"feature": feature, "logistic_p05_threshold": p05, "n": len(tmp)}])


def plot_coupling(df: pd.DataFrame):
    xvar = "I_static_toughness"
    yvar = "delta_gamma_excess_pre_drop"
    if xvar not in df.columns or yvar not in df.columns:
        return
    tmp = df[["sigma", "cov", "IG_label", xvar, yvar]].copy()
    tmp[xvar] = to_numeric_series(tmp[xvar])
    tmp[yvar] = to_numeric_series(tmp[yvar])
    tmp["IG_label"] = to_numeric_series(tmp["IG_label"])
    tmp = tmp[np.isfinite(tmp[xvar]) & np.isfinite(tmp[yvar])]
    if tmp.empty:
        return

    colors = {"sigma3": "tab:blue", "sigma5": "tab:orange", "sigma11": "tab:green", "sigma17": "tab:purple"}
    markers = {0: "o", 1: "o"}
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for sigma in SIGMA_ORDER:
        for lab in [0, 1]:
            g = tmp[(tmp["sigma"] == sigma) & (tmp["IG_label"] == lab)]
            if g.empty:
                continue
            ax.scatter(g[xvar], g[yvar], s=70,
                       marker=markers[lab],
                       facecolors=colors[sigma] if lab == 1 else "none",
                       edgecolors=colors[sigma], linewidths=1.4,
                       label=f"{pretty_sigma(sigma)} / IG={lab}")
    for _, r in tmp.iterrows():
        if r["cov"] in ["0H", "025cov", "050cov", "100cov"]:
            ax.text(r[xvar], r[yvar], "  " + pretty_cov(r["cov"]), fontsize=7)
    ax.set_xlabel(pretty_feature(xvar))
    ax.set_ylabel(pretty_feature(yvar))
    ax.legend(frameon=False, loc="bestoutside" if False else "best", fontsize=8)
    ax.tick_params(direction="in")
    ax.grid(False)
    bottom_title(fig, "(e) Static toughness degradation vs dynamic shear localization", y=0.005)
    fig.subplots_adjust(bottom=0.17)
    save_fig(fig, FIG_DIR / "fig_static_dynamic_coupling")


# =============================================================================
# 6. 主流程
# =============================================================================

def main():
    print("=" * 90)
    print("机器学习辅助的可解释阈值识别开始")
    print(f"BASE_DIR: {BASE_DIR}")
    print(f"OUT_DIR : {OUT_DIR}")
    print(f"sklearn available: {SKLEARN_OK}")
    print("=" * 90)

    if not KEY_METRICS_CSV.exists():
        raise FileNotFoundError(f"找不到 key_metrics_summary.csv: {KEY_METRICS_CSV}")
    if not ALL_CASES_CSV.exists():
        raise FileNotFoundError(f"找不到 all_cases_summary.csv: {ALL_CASES_CSV}")

    key_df = pd.read_csv(KEY_METRICS_CSV, encoding="utf-8-sig")
    all_df = pd.read_csv(ALL_CASES_CSV, encoding="utf-8-sig")
    key_features = build_key_feature_table(key_df)
    curve_features = build_full_curve_feature_table(all_df)
    static_mech = load_static_mechanical()
    labels = load_labels()

    feature_table = merge_tables([key_features, curve_features, static_mech, labels])
    feature_table.to_csv(OUT_FEATURE_CSV, index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(OUT_FEATURE_XLSX) as writer:
        feature_table.to_excel(writer, sheet_name="features", index=False)

    print(f"特征总表：{OUT_FEATURE_CSV}")
    print(f"特征总表 Excel：{OUT_FEATURE_XLSX}")
    if labels.empty:
        print("WARNING: 没有找到 IG_label 标签文件。只生成特征表，不做阈值/ML 分析。")
        return

    ml_df = make_ml_dataset(feature_table)
    if ml_df.empty:
        raise ValueError("ML 数据为空。请检查 IG_label 是否存在且为 0/1。")

    available_features = [f for f in CANDIDATE_FEATURES if f in ml_df.columns]
    for f in available_features:
        ml_df[f] = to_numeric_series(ml_df[f])
    print(f"有效样本数：{len(ml_df)}")
    print(f"候选特征数：{len(available_features)}")

    all_scan, best = find_best_threshold(ml_df, available_features)
    if best.empty:
        raise ValueError("没有任何候选特征可完成阈值扫描。请检查特征列和 IG_label。")

    best_row = best.iloc[0]
    if PREFERRED_PRIMARY_FEATURE in best["feature"].values:
        primary_row = best[best["feature"] == PREFERRED_PRIMARY_FEATURE].iloc[0]
    else:
        primary_row = best_row
    primary_feature = str(primary_row["feature"])

    # 按全样本最优阈值给每个样本加预测列
    ml_df["best_feature_pred"] = apply_threshold(
        ml_df[best_row["feature"]], float(best_row["threshold"]), str(best_row["direction"])
    )
    ml_df["primary_feature_pred"] = apply_threshold(
        ml_df[primary_feature], float(primary_row["threshold"]), str(primary_row["direction"])
    )

    # LOOCV
    loocv_adaptive = loocv_threshold(ml_df, available_features, fixed_feature=None)
    loocv_primary = loocv_threshold(ml_df, available_features, fixed_feature=primary_feature)
    loocv_adaptive_summary = summarize_loocv(loocv_adaptive)
    loocv_primary_summary = summarize_loocv(loocv_primary)

    # Bootstrap
    boot = pd.DataFrame()
    boot_summary = pd.DataFrame()
    if DO_BOOTSTRAP:
        boot = bootstrap_best_threshold(ml_df, primary_feature, N_BOOTSTRAP)
        if not boot.empty:
            boot_summary = pd.DataFrame([{
                "feature": primary_feature,
                "n_bootstrap_success": len(boot),
                "threshold_median": float(np.nanmedian(boot["threshold"])),
                "threshold_ci2p5": float(np.nanpercentile(boot["threshold"], 2.5)),
                "threshold_ci97p5": float(np.nanpercentile(boot["threshold"], 97.5)),
                "f1_median": float(np.nanmedian(boot["f1"])),
                "accuracy_median": float(np.nanmedian(boot["accuracy"])),
            }])

    # Logistic
    logistic_summary = plot_logistic(ml_df, primary_feature)

    # 绘图
    plot_feature_rank(best)
    plot_best_feature_threshold(ml_df, best_row)
    plot_roc(ml_df, best_row)
    plot_coupling(ml_df)

    # 汇总文字
    summary_lines = []
    summary_lines.append("机器学习辅助的可解释阈值识别结果")
    summary_lines.append("=" * 60)
    summary_lines.append(f"样本数: {len(ml_df)}")
    summary_lines.append(f"候选特征数: {len(available_features)}")
    summary_lines.append("")
    summary_lines.append("全样本最优单指标阈值：")
    summary_lines.append(f"  feature   = {best_row['feature']}")
    summary_lines.append(f"  threshold = {best_row['threshold']}")
    summary_lines.append(f"  direction = {best_row['direction']}")
    summary_lines.append(f"  F1        = {best_row['f1']:.4f}")
    summary_lines.append(f"  Accuracy  = {best_row['accuracy']:.4f}")
    summary_lines.append(f"  AUC       = {best_row['auc']:.4f}")
    summary_lines.append("")
    summary_lines.append("论文主推荐特征/或优先动态局部化特征：")
    summary_lines.append(f"  feature   = {primary_feature}")
    summary_lines.append(f"  threshold = {primary_row['threshold']}")
    summary_lines.append(f"  direction = {primary_row['direction']}")
    summary_lines.append(f"  F1        = {primary_row['f1']:.4f}")
    summary_lines.append(f"  Accuracy  = {primary_row['accuracy']:.4f}")
    summary_lines.append(f"  AUC       = {primary_row['auc']:.4f}")
    summary_lines.append("")
    summary_lines.append("LOOCV adaptive stump:")
    for k, v in loocv_adaptive_summary.items():
        summary_lines.append(f"  {k}: {v}")
    summary_lines.append("")
    summary_lines.append(f"LOOCV fixed primary feature ({primary_feature}):")
    for k, v in loocv_primary_summary.items():
        summary_lines.append(f"  {k}: {v}")
    if not boot_summary.empty:
        summary_lines.append("")
        summary_lines.append("Bootstrap threshold CI:")
        for k, v in boot_summary.iloc[0].items():
            summary_lines.append(f"  {k}: {v}")

    summary_txt = OUT_DIR / "ml_summary.txt"
    summary_txt.write_text("\n".join(summary_lines), encoding="utf-8-sig")

    # 输出 Excel
    with pd.ExcelWriter(OUT_ANALYSIS_XLSX) as writer:
        feature_table.to_excel(writer, sheet_name="feature_table_all", index=False)
        ml_df.to_excel(writer, sheet_name="ml_samples", index=False)
        all_scan.to_excel(writer, sheet_name="threshold_scan_all", index=False)
        best.to_excel(writer, sheet_name="best_single_features", index=False)
        loocv_adaptive.to_excel(writer, sheet_name="LOOCV_adaptive_stump", index=False)
        loocv_primary.to_excel(writer, sheet_name="LOOCV_primary_feature", index=False)
        pd.DataFrame([loocv_adaptive_summary]).to_excel(writer, sheet_name="LOOCV_adaptive_summary", index=False)
        pd.DataFrame([loocv_primary_summary]).to_excel(writer, sheet_name="LOOCV_primary_summary", index=False)
        if not boot.empty:
            boot.to_excel(writer, sheet_name="bootstrap_thresholds", index=False)
        if not boot_summary.empty:
            boot_summary.to_excel(writer, sheet_name="bootstrap_summary", index=False)
        if not logistic_summary.empty:
            logistic_summary.to_excel(writer, sheet_name="logistic_summary", index=False)

    # 单独 CSV 输出
    all_scan.to_csv(OUT_DIR / "threshold_scan_all.csv", index=False, encoding="utf-8-sig")
    best.to_csv(OUT_DIR / "best_single_features.csv", index=False, encoding="utf-8-sig")
    loocv_adaptive.to_csv(OUT_DIR / "loocv_adaptive_stump.csv", index=False, encoding="utf-8-sig")
    loocv_primary.to_csv(OUT_DIR / "loocv_primary_feature.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 90)
    print("完成。主要输出：")
    print(f"  {OUT_ANALYSIS_XLSX}")
    print(f"  {OUT_FEATURE_XLSX}")
    print(f"  {summary_txt}")
    print(f"  figures: {FIG_DIR}")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()
