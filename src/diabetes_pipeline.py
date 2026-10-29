"""Reproducible, non-clinical pipeline for NHANES diabetes status research.

The target is existing self-reported doctor-diagnosed diabetes status
(`DIQ010`), not future diabetes incidence.
"""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


FEATURES = [
    "RIDAGEYR",
    "RIAGENDR",
    "BMXBMI",
    "BMXWAIST",
    "systolic_avg",
    "diastolic_avg",
]
TARGET = "diabetes"
PARTICIPANT_ID = "SEQN"
TARGET_DEFINITION = (
    "Existing self-reported doctor-diagnosed diabetes status: "
    "DIQ010=1 is positive; DIQ010=2 is negative; all other or missing "
    "responses are excluded. This is not future diabetes incidence."
)
SPLIT_SEED = 2026
THRESHOLDS = tuple(round(float(x), 2) for x in np.arange(0.10, 0.501, 0.05))
RF_COMMON_PARAMS = {
    "n_estimators": 300,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}
RF_CONFIGURATIONS = (1, 5)


@dataclass(frozen=True)
class DataPreparation:
    data: pd.DataFrame
    counts: dict[str, int]


@dataclass(frozen=True)
class DataSplit:
    X: pd.DataFrame
    y: pd.Series
    participant_ids: pd.Series


@dataclass(frozen=True)
class DataSplits:
    train: DataSplit
    validation: DataSplit
    test: DataSplit
    counts: dict[str, dict[str, int]]
    seed: int = SPLIT_SEED


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def _validate_participant_ids(frame: pd.DataFrame, name: str) -> None:
    _require_columns(frame, {PARTICIPANT_ID}, name)
    ids = frame[PARTICIPANT_ID]
    if ids.isna().any():
        raise ValueError(f"{name}.{PARTICIPANT_ID} contains missing IDs")
    if ids.duplicated().any():
        raise ValueError(f"{name}.{PARTICIPANT_ID} contains duplicate IDs")
    if not pd.api.types.is_numeric_dtype(ids) or pd.api.types.is_bool_dtype(ids):
        raise TypeError(f"{name}.{PARTICIPANT_ID} must be numeric")
    numeric_ids = ids.to_numpy(dtype=float)
    if not np.isfinite(numeric_ids).all():
        raise ValueError(f"{name}.{PARTICIPANT_ID} must be finite")
    if not np.equal(numeric_ids, np.floor(numeric_ids)).all():
        raise ValueError(f"{name}.{PARTICIPANT_ID} must contain whole-number IDs")
    if (numeric_ids <= 0).any():
        raise ValueError(f"{name}.{PARTICIPANT_ID} must be positive")


def _require_numeric(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    _require_columns(frame, set(columns), name)
    non_numeric = [c for c in columns if not pd.api.types.is_numeric_dtype(frame[c])]
    if non_numeric:
        raise TypeError(f"{name} columns must be numeric: {non_numeric}")


def _validate_join_inputs(
    frames: dict[str, pd.DataFrame], requirements: dict[str, set[str]]
) -> None:
    for name, frame in frames.items():
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"{name} must be a pandas DataFrame")
        _validate_participant_ids(frame, name)
        _require_columns(frame, requirements[name], name)


def prepare_frames(
    demo: pd.DataFrame,
    body: pd.DataFrame,
    diabetes: pd.DataFrame,
    blood_pressure: pd.DataFrame,
) -> DataPreparation:
    """Prepare adult complete cases from already-read NHANES component frames."""
    frames = {
        "DEMO": demo,
        "BMX": body,
        "DIQ": diabetes,
        "BPXO": blood_pressure,
    }
    required = {
        "DEMO": {"RIDAGEYR", "RIAGENDR"},
        "BMX": {"BMXBMI", "BMXWAIST"},
        "DIQ": {"DIQ010"},
        "BPXO": {
            "BPXOSY1", "BPXOSY2", "BPXOSY3",
            "BPXODI1", "BPXODI2", "BPXODI3",
        },
    }
    _validate_join_inputs(frames, required)
    _require_numeric(demo, ["RIDAGEYR", "RIAGENDR"], "DEMO")
    _require_numeric(body, ["BMXBMI", "BMXWAIST"], "BMX")
    _require_numeric(diabetes, ["DIQ010"], "DIQ")
    bp_cols = [
        "BPXOSY1", "BPXOSY2", "BPXOSY3",
        "BPXODI1", "BPXODI2", "BPXODI3",
    ]
    _require_numeric(blood_pressure, bp_cols, "BPXO")

    # Preserve only the variables used by the recorded diabetes workflow.
    joined = demo[[PARTICIPANT_ID, "RIDAGEYR", "RIAGENDR"]].merge(
        body[[PARTICIPANT_ID, "BMXBMI", "BMXWAIST"]],
        on=PARTICIPANT_ID,
        how="inner",
        validate="one_to_one",
    )
    joined = joined.merge(
        diabetes[[PARTICIPANT_ID, "DIQ010"]],
        on=PARTICIPANT_ID,
        how="inner",
        validate="one_to_one",
    )

    counts = {"demo_rows": int(len(demo)), "demographics_body_diabetes_rows": int(len(joined))}
    valid_target = joined["DIQ010"].isin([1.0, 2.0])
    counts["excluded_target_response_rows"] = int((~valid_target).sum())
    joined = joined.loc[valid_target].copy()
    joined[TARGET] = (joined["DIQ010"] == 1.0).astype("int8")

    adult = joined["RIDAGEYR"] >= 18
    counts["excluded_under_18_rows"] = int((~adult).sum())
    joined = joined.loc[adult].copy()

    bp = blood_pressure[[PARTICIPANT_ID, *bp_cols]].copy()
    bp["systolic_avg"] = bp[["BPXOSY1", "BPXOSY2", "BPXOSY3"]].mean(axis=1)
    bp["diastolic_avg"] = bp[["BPXODI1", "BPXODI2", "BPXODI3"]].mean(axis=1)
    bp = bp[[PARTICIPANT_ID, "systolic_avg", "diastolic_avg"]]
    joined = joined.merge(bp, on=PARTICIPANT_ID, how="left", validate="one_to_one")

    selected = joined[[PARTICIPANT_ID, *FEATURES, TARGET]].copy()
    _require_numeric(selected, FEATURES + [TARGET], "prepared data")
    selected.loc[:, FEATURES] = selected[FEATURES].replace([np.inf, -np.inf], np.nan)
    complete = selected[FEATURES].notna().all(axis=1)
    counts["excluded_incomplete_feature_rows"] = int((~complete).sum())
    selected = selected.loc[complete].sort_values(PARTICIPANT_ID).reset_index(drop=True)
    if selected.empty:
        raise ValueError("No adult complete cases remain after preparation")
    if selected[PARTICIPANT_ID].duplicated().any():
        raise ValueError("Prepared data contains duplicate participant IDs")
    if selected[FEATURES].isna().any().any():
        raise ValueError("Prepared feature data contains missing values")
    counts["adult_complete_case_rows"] = int(len(selected))
    counts["positive_target_rows"] = int(selected[TARGET].sum())
    counts["negative_target_rows"] = int((selected[TARGET] == 0).sum())
    return DataPreparation(selected, counts)


def load_nhanes_data(data_dir: str | Path) -> DataPreparation:
    """Load the four component files required for the recorded feature set."""
    root = Path(data_dir)
    filenames = {
        "DEMO": "DEMO_L.xpt",
        "BMX": "BMX_L.xpt",
        "DIQ": "DIQ_L.xpt",
        "BPXO": "BPXO_L.xpt",
    }
    paths = {key: root / filename for key, filename in filenames.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required NHANES files: {missing}")
    frames = {key: pd.read_sas(path, format="xport") for key, path in paths.items()}
    return prepare_frames(frames["DEMO"], frames["BMX"], frames["DIQ"], frames["BPXO"])


def summarize_diabetes_data(data: pd.DataFrame, bins: int = 10) -> dict[str, Any]:
    """Create descriptive class, feature, and within-class distribution summaries.

    This summarizes the prepared adult complete-case dataset. It does not
    estimate causal effects or fit a predictive model.
    """
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
        raise ValueError("bins must be a positive integer")
    required = {TARGET, *FEATURES}
    _require_columns(data, required, "statistical analysis data")
    _require_numeric(data, FEATURES + [TARGET], "statistical analysis data")
    if data.empty or data[TARGET].isna().any() or not set(data[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Statistical analysis requires non-empty data with binary target values 0 and 1")

    status_labels = {0: "No reported diabetes", 1: "Reported diabetes"}
    class_counts = (
        data[TARGET].astype(int).value_counts().reindex([0, 1], fill_value=0)
        .rename(index=status_labels).rename("participants").to_frame()
    )
    class_counts["percent"] = class_counts["participants"] / len(data) * 100

    analysis_features = [
        "RIDAGEYR", "BMXBMI", "BMXWAIST", "systolic_avg", "diastolic_avg"
    ]
    rows = []
    for status, subset in [("All participants", data), *[(label, data.loc[data[TARGET] == value]) for value, label in status_labels.items()]]:
        for feature in analysis_features:
            values = subset[feature].dropna().astype(float)
            rows.append({
                "status": status,
                "feature": feature,
                "n": int(values.size),
                "mean": float(values.mean()) if values.size else float("nan"),
                "median": float(values.median()) if values.size else float("nan"),
                "standard_deviation": float(values.std(ddof=1)) if values.size > 1 else float("nan"),
            })
    descriptive_statistics = pd.DataFrame(rows)

    distributions = {}
    for feature in analysis_features:
        values = data[feature].astype(float)
        intervals = pd.cut(values, bins=bins, include_lowest=True, duplicates="drop")
        counts = pd.crosstab(intervals, data[TARGET].astype(int)).reindex(columns=[0, 1], fill_value=0)
        totals = counts.sum(axis=0).replace(0, np.nan)
        percentages = counts.div(totals, axis=1).mul(100).fillna(0)
        percentages.columns = [status_labels[int(value)] for value in percentages.columns]
        percentages.index = percentages.index.astype(str)
        percentages.index.name = "Range"
        distributions[feature] = percentages

    return {
        "participant_count": int(len(data)),
        "class_distribution": class_counts,
        "descriptive_statistics": descriptive_statistics,
        "distributions": distributions,
    }


def build_evaluation_comparison(
    original_evaluation: dict[str, Any],
    candidate_evaluation: dict[str, Any],
    candidate_metadata: dict[str, Any],
) -> pd.DataFrame:
    """Build comparable saved-score rows without loading either model artifact."""
    if original_evaluation.get("test_samples") != candidate_evaluation.get("test_samples"):
        raise ValueError("Model evaluations use different participant counts and cannot be compared directly")
    parameters = candidate_metadata.get("model_parameters")
    if not isinstance(parameters, dict) or "min_samples_leaf" not in parameters:
        raise ValueError("Candidate metadata is missing min_samples_leaf")
    candidate_label = (
        "Stage 1 candidate "
        f"(min_samples_leaf={parameters['min_samples_leaf']}; not used by app)"
    )
    original_label = "Original app model (currently used)"
    metric_sources = {
        "Precision": ("precision", "precision"),
        "Recall": ("recall", "recall"),
        "F1-score": ("f1_score", "f1"),
        "ROC-AUC": ("roc_auc", "roc_auc"),
    }
    rows = {}
    for label, original_key, candidate_key in (
        (metric, keys[0], keys[1]) for metric, keys in metric_sources.items()
    ):
        try:
            original_value = float(original_evaluation[original_key])
            candidate_value = float(candidate_evaluation[candidate_key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Saved comparison metric is missing or invalid: {label}") from exc
        if not np.isfinite([original_value, candidate_value]).all() or not (
            0 <= original_value <= 1 and 0 <= candidate_value <= 1
        ):
            raise ValueError(f"Saved comparison metric must be finite and between 0 and 1: {label}")
        rows[label] = {original_label: original_value, candidate_label: candidate_value}
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("Metric")


def validate_feature_frame(X: pd.DataFrame) -> None:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("Model features must be a pandas DataFrame")
    if list(X.columns) != FEATURES:
        raise ValueError(f"Feature names/order must be exactly {FEATURES}")
    _require_numeric(X, FEATURES, "model features")
    values = X.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Model features must contain only finite numeric values")


def build_prediction_input(
    age: int | float,
    sex: str,
    bmi: int | float,
    waist_cm: int | float,
    systolic_mm_hg: int | float,
    diastolic_mm_hg: int | float,
) -> pd.DataFrame:
    """Map app values to the fitted NHANES feature schema without Streamlit."""
    if sex not in {"Male", "Female"}:
        raise ValueError("sex must be 'Male' or 'Female'")
    vals = [age, 1 if sex == "Male" else 2, bmi, waist_cm, systolic_mm_hg, diastolic_mm_hg]
    try:
        numeric = np.asarray(vals, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError("All model inputs must be numeric") from exc
    if not np.isfinite(numeric).all():
        raise ValueError("All model inputs must be finite")
    frame = pd.DataFrame([vals], columns=FEATURES)
    validate_feature_frame(frame)
    return frame


def average_blood_pressure_readings(
    systolic_readings: list[int | float | None] | tuple[int | float | None, ...],
    diastolic_readings: list[int | float | None] | tuple[int | float | None, ...],
) -> tuple[float | None, float | None]:
    """Average available readings independently; absent values are never treated as zero.

    A missing average is returned as ``None``. The caller must then avoid
    classification until at least one reading of that pressure is available.
    """

    def average(values, label, lower, upper):
        available = []
        for index, value in enumerate(values, start=1):
            if value is None:
                continue
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
                raise TypeError(f"{label} reading {index} must be numeric or None")
            number = float(value)
            if not np.isfinite(number):
                raise ValueError(f"{label} reading {index} must be finite")
            if not lower <= number <= upper:
                raise ValueError(f"{label} reading {index} must be between {lower} and {upper} mmHg")
            available.append(number)
        return float(np.mean(available)) if available else None

    systolic_average = average(systolic_readings, "Systolic", 60, 250)
    diastolic_average = average(diastolic_readings, "Diastolic", 30, 150)
    return systolic_average, diastolic_average


def make_splits(data: pd.DataFrame, seed: int = SPLIT_SEED) -> DataSplits:
    """Create stratified 60/20/20 splits and verify ID separation."""
    required = {PARTICIPANT_ID, TARGET, *FEATURES}
    _require_columns(data, required, "prepared data")
    if list(c for c in data.columns if c in FEATURES) != FEATURES:
        raise ValueError("Prepared feature order is invalid")
    _validate_participant_ids(data, "prepared data")
    validate_feature_frame(data[FEATURES])
    if data[TARGET].isna().any() or set(data[TARGET].unique()) != {0, 1}:
        raise ValueError("Target must contain both binary classes 0 and 1")

    train_df, temporary = train_test_split(
        data, test_size=0.40, random_state=seed, stratify=data[TARGET]
    )
    validation_df, test_df = train_test_split(
        temporary, test_size=0.50, random_state=seed, stratify=temporary[TARGET]
    )

    def as_split(frame: pd.DataFrame) -> DataSplit:
        return DataSplit(
            X=frame[FEATURES].reset_index(drop=True),
            y=frame[TARGET].astype(int).reset_index(drop=True),
            participant_ids=frame[PARTICIPANT_ID].reset_index(drop=True),
        )

    result = DataSplits(as_split(train_df), as_split(validation_df), as_split(test_df), {}, seed)
    verify_split_integrity(result)
    counts: dict[str, dict[str, int]] = {}
    for name, split in (("train", result.train), ("validation", result.validation), ("test", result.test)):
        counts[name] = {
            "rows": int(len(split.y)),
            "positive": int(split.y.sum()),
            "negative": int((split.y == 0).sum()),
        }
    return DataSplits(result.train, result.validation, result.test, counts, seed)


def verify_split_integrity(splits: DataSplits) -> dict[str, int]:
    named = {
        "train": splits.train.participant_ids,
        "validation": splits.validation.participant_ids,
        "test": splits.test.participant_ids,
    }
    for name, ids in named.items():
        if ids.isna().any() or ids.duplicated().any():
            raise ValueError(f"{name} participant IDs are missing or duplicated")
    overlaps = {
        "train_validation": len(set(named["train"]) & set(named["validation"])),
        "train_test": len(set(named["train"]) & set(named["test"])),
        "validation_test": len(set(named["validation"]) & set(named["test"])),
    }
    if any(overlaps.values()):
        raise ValueError(f"Participant IDs overlap across splits: {overlaps}")
    return overlaps


def split_manifest(splits: DataSplits) -> dict[str, Any]:
    """Describe split membership without persisting participant IDs."""
    records = {}
    for name, split in (("train", splits.train), ("validation", splits.validation), ("test", splits.test)):
        normalized_ids = sorted(str(value) for value in split.participant_ids.tolist())
        digest = hashlib.sha256("\n".join(normalized_ids).encode("utf-8")).hexdigest()
        records[name] = {"rows": len(normalized_ids), "participant_id_sha256": digest}
    return {"seed": splits.seed, "proportions": {"train": 0.60, "validation": 0.20, "test": 0.20}, "membership": records}


def _new_model(min_samples_leaf: int) -> RandomForestClassifier:
    if min_samples_leaf not in RF_CONFIGURATIONS:
        raise ValueError(f"Unsupported min_samples_leaf: {min_samples_leaf}")
    return RandomForestClassifier(min_samples_leaf=min_samples_leaf, **RF_COMMON_PARAMS)


def _positive_probability(model: RandomForestClassifier, X: pd.DataFrame) -> np.ndarray:
    validate_feature_frame(X)
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("Fitted model lacks positive class 1")
    return model.predict_proba(X)[:, classes.index(1)]


def validate_model_metadata(model: Any, metadata: dict[str, Any]) -> None:
    """Check persisted estimator schema and identity fields before use."""
    if not isinstance(model, RandomForestClassifier):
        raise TypeError("Candidate estimator must be a RandomForestClassifier")
    if not isinstance(metadata, dict):
        raise TypeError("Model metadata must be a JSON object")
    if list(getattr(model, "classes_", [])) != [0, 1]:
        raise ValueError("Fitted model must contain classes 0 and 1")
    if list(getattr(model, "feature_names_in_", [])) != metadata.get("features"):
        raise ValueError("Fitted feature names/order do not match metadata")
    if getattr(model, "n_features_in_", None) != len(FEATURES):
        raise ValueError("Fitted feature count does not match the pipeline schema")
    if metadata.get("features") != FEATURES:
        raise ValueError("Metadata features/order do not match the pipeline schema")
    threshold = metadata.get("threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not 0 < threshold < 1:
        raise ValueError("Metadata threshold must be numeric and strictly between 0 and 1")
    if metadata.get("model_type") != type(model).__name__:
        raise ValueError("Estimator type does not match metadata")
    if metadata.get("model_parameters") != model.get_params(deep=False):
        raise ValueError("Estimator parameters do not match metadata")
    if metadata.get("target") != TARGET_DEFINITION:
        raise ValueError("Metadata target definition does not match the pipeline target")


def calibration_diagnostics(y_true: pd.Series, probabilities: np.ndarray, n_bins: int = 10) -> dict[str, Any]:
    """Return descriptive quantile-bin calibration data; does not fit a calibrator."""
    observed, predicted = calibration_curve(
        y_true, probabilities, n_bins=min(n_bins, max(2, len(y_true) // 20)), strategy="quantile"
    )
    bin_rows = [
        {"mean_predicted_probability": float(p), "observed_positive_rate": float(o)}
        for p, o in zip(predicted, observed)
    ]
    ece = float(np.mean([abs(row["mean_predicted_probability"] - row["observed_positive_rate"]) for row in bin_rows]))
    return {"method": "quantile-binned calibration curve; descriptive, no calibrator fitted", "bins": bin_rows, "mean_absolute_bin_gap": ece}


def _threshold_metrics(y_true: pd.Series, probabilities: np.ndarray, threshold: float) -> dict[str, Any]:
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = (int(x) for x in matrix.ravel())
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "confusion_matrix": matrix.tolist(),
    }


def _select_threshold(y_validation: pd.Series, probabilities: np.ndarray) -> tuple[float, dict[str, Any]]:
    rows = [_threshold_metrics(y_validation, probabilities, threshold) for threshold in THRESHOLDS]
    # Maximize validation F1; ties prefer precision, then the higher threshold.
    selected = max(rows, key=lambda row: (row["f1"], row["precision"], row["threshold"]))
    return selected["threshold"], {
        "criterion": "maximum validation F1 over thresholds 0.10 through 0.50 in 0.05 increments",
        "tie_break": "higher validation precision, then higher threshold",
        "selected_validation_metrics": selected,
        "threshold_grid": list(THRESHOLDS),
    }


def select_model_and_threshold(splits: DataSplits) -> tuple[RandomForestClassifier, dict[str, Any]]:
    """Fit both candidates on training only; select model and threshold on validation."""
    candidate_results = []
    fitted = {}
    for leaf in RF_CONFIGURATIONS:
        model = _new_model(leaf)
        model.fit(splits.train.X, splits.train.y)
        probabilities = _positive_probability(model, splits.validation.X)
        threshold, selection = _select_threshold(splits.validation.y, probabilities)
        candidate_results.append({
            "min_samples_leaf": leaf,
            "parameters": model.get_params(deep=False),
            "validation_roc_auc": float(roc_auc_score(splits.validation.y, probabilities)),
            "validation_brier_score": float(brier_score_loss(splits.validation.y, probabilities)),
            "validation_calibration": calibration_diagnostics(splits.validation.y, probabilities),
            "threshold_selection": selection,
        })
        fitted[leaf] = model

    # Select configuration by validation F1; deterministic tie-breaks match threshold selection.
    chosen = max(
        candidate_results,
        key=lambda row: (
            row["threshold_selection"]["selected_validation_metrics"]["f1"],
            row["threshold_selection"]["selected_validation_metrics"]["precision"],
            row["threshold_selection"]["selected_validation_metrics"]["threshold"],
            -row["min_samples_leaf"],
        ),
    )
    selected_leaf = chosen["min_samples_leaf"]
    details = {
        "selected_min_samples_leaf": selected_leaf,
        "selected_parameters": fitted[selected_leaf].get_params(deep=False),
        "threshold": chosen["threshold_selection"]["selected_validation_metrics"]["threshold"],
        "threshold_selection": chosen["threshold_selection"],
        "configuration_selection_criterion": "highest validation F1 at each configuration's validation-selected threshold",
        "candidate_results": candidate_results,
    }
    return fitted[selected_leaf], details


def evaluate_development_test(
    model: RandomForestClassifier, splits: DataSplits, threshold: float
) -> dict[str, Any]:
    """Evaluate the previously examined participant set as development data."""
    probabilities = _positive_probability(model, splits.test.X)
    threshold_stats = _threshold_metrics(splits.test.y, probabilities, threshold)
    return {
        "evaluation_label": "development evaluation; this test participant set was previously examined",
        "target_definition": TARGET_DEFINITION,
        "test_samples": int(len(splits.test.y)),
        "roc_auc": float(roc_auc_score(splits.test.y, probabilities)),
        "brier_score": float(brier_score_loss(splits.test.y, probabilities)),
        "precision": threshold_stats["precision"],
        "recall": threshold_stats["recall"],
        "specificity": threshold_stats["specificity"],
        "f1": threshold_stats["f1"],
        "accuracy": threshold_stats["accuracy"],
        "threshold": float(threshold),
        "confusion_matrix": threshold_stats["confusion_matrix"],
        "calibration": calibration_diagnostics(splits.test.y, probabilities),
        "calibration_limit": "descriptive only; no calibration model was fit on this test set",
    }


def build_run_records(prepared: DataPreparation, splits: DataSplits, model: RandomForestClassifier, selection: dict[str, Any], evaluation: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = {
        "model_type": type(model).__name__,
        "model_parameters": model.get_params(deep=False),
        "features": FEATURES,
        "target": TARGET_DEFINITION,
        "dataset": "NHANES 2021-2023 local XPT files",
        "preparation_counts": prepared.counts,
        "split_seed": splits.seed,
        "split_counts": splits.counts,
        "participant_split_integrity": verify_split_integrity(splits),
        "split_manifest": split_manifest(splits),
        "threshold": selection["threshold"],
        "threshold_selection": selection["threshold_selection"],
        "configuration_selection_criterion": selection["configuration_selection_criterion"],
        "evaluation_method": "model/configuration and threshold selected on training/validation; test evaluated once as development data",
    }
    return metadata, evaluation


def write_candidate_artifacts(
    model: RandomForestClassifier,
    metadata: dict[str, Any],
    evaluation: dict[str, Any],
    output_dir: str | Path,
    stem: str = "diabetes_status_candidate_stage1",
) -> list[Path]:
    """Write new, distinct artifacts; refuse to overwrite any existing path."""
    root = Path(output_dir)
    paths = [root / f"{stem}.joblib", root / f"{stem}_metadata.json", root / f"{stem}_evaluation.json"]
    validate_model_metadata(model, metadata)
    if any(path.exists() for path in paths):
        raise FileExistsError(f"Candidate artifact path already exists; refusing overwrite: {paths}")
    root.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, paths[0])
    paths[1].write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
    paths[2].write_text(json.dumps(evaluation, indent=2, allow_nan=False), encoding="utf-8")
    return paths


def run_pipeline(data_dir: str | Path, output_dir: str | Path, stem: str = "diabetes_status_candidate_stage1") -> dict[str, Any]:
    """Run preparation, development selection/evaluation, then save candidate artifacts."""
    prepared = load_nhanes_data(data_dir)
    splits = make_splits(prepared.data)
    model, selection = select_model_and_threshold(splits)
    evaluation = evaluate_development_test(model, splits, selection["threshold"])
    metadata, evaluation = build_run_records(prepared, splits, model, selection, evaluation)
    data_root = Path(data_dir)
    source_files = ["DEMO_L.xpt", "BMX_L.xpt", "DIQ_L.xpt", "BPXO_L.xpt"]
    metadata["source_files"] = {
        name: hashlib.sha256((data_root / name).read_bytes()).hexdigest()
        for name in source_files
    }
    paths = write_candidate_artifacts(model, metadata, evaluation, output_dir, stem=stem)
    return {"prepared": prepared, "splits": splits, "selection": selection, "evaluation": evaluation, "metadata": metadata, "artifact_paths": paths}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--stem", default="diabetes_status_candidate_stage1")
    args = parser.parse_args()
    result = run_pipeline(args.data_dir, args.output_dir, stem=args.stem)
    print(json.dumps({
        "preparation_counts": result["prepared"].counts,
        "split_counts": result["splits"].counts,
        "overlaps": result["metadata"]["participant_split_integrity"],
        "selection": result["selection"],
        "evaluation": result["evaluation"],
        "artifact_paths": [str(p) for p in result["artifact_paths"]],
    }, indent=2, allow_nan=False))
