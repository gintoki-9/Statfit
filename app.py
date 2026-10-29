import json
from numbers import Real
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
from src.diabetes_pipeline import (
    average_blood_pressure_readings,
    build_prediction_input,
    load_nhanes_data,
    summarize_diabetes_data,
    build_evaluation_comparison,
    FEATURES,
    RF_CONFIGURATIONS,
    TARGET_DEFINITION,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="StatFit",
    page_icon="💪",
    layout="wide"
)

PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODELS_DIR / "diabetes_status_random_forest.joblib"
METADATA_PATH = MODELS_DIR / "diabetes_status_metadata.json"
EVALUATION_PATH = MODELS_DIR / "diabetes_status_evaluation.json"
STAGE1_CANDIDATE_EVALUATION_PATH = MODELS_DIR / "diabetes_status_candidate_stage1_manifest_evaluation.json"
PIPELINE_METADATA_PATH = MODELS_DIR / "diabetes_status_candidate_stage1_manifest_metadata.json"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

@st.cache_resource
def load_model():
    """Load the saved model and its metadata."""

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    if not METADATA_PATH.is_file():
        raise FileNotFoundError(
            f"Model metadata not found: {METADATA_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    validate_model_and_metadata(model, metadata)

    return model, metadata


def validate_model_and_metadata(model, metadata):
    """Reject incompatible model artifacts before they can predict."""

    if not isinstance(model, RandomForestClassifier):
        raise TypeError(
            "The saved estimator must be a scikit-learn "
            "RandomForestClassifier."
        )

    classes = list(getattr(model, "classes_", []))
    if len(classes) != 2 or 0 not in classes or 1 not in classes:
        raise ValueError(
            "The fitted model must contain exactly the diabetes "
            "status classes 0 and 1."
        )

    if not isinstance(metadata, dict):
        raise ValueError("The model metadata must be a JSON object.")

    expected_features = metadata.get("features")
    if not isinstance(expected_features, list) or not all(
        isinstance(feature, str) for feature in expected_features
    ):
        raise ValueError(
            "Model metadata must contain a list of feature names."
        )

    fitted_features = getattr(model, "feature_names_in_", None)
    if fitted_features is None or list(fitted_features) != expected_features:
        raise ValueError(
            "The fitted model feature names and order do not match "
            "the metadata feature list."
        )

    fitted_feature_count = getattr(model, "n_features_in_", None)
    if fitted_feature_count != len(expected_features):
        raise ValueError(
            "The fitted model feature count does not match the "
            "metadata feature list."
        )

    threshold = metadata.get("threshold")
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, Real)
        or not 0 < float(threshold) < 1
    ):
        raise ValueError(
            "The model metadata threshold must be a number strictly "
            "between 0 and 1."
        )


@st.cache_data
def load_json_report(path: str):
    """Read a local JSON report without loading serialized model artifacts."""
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data(show_spinner="Preparing NHANES descriptive summaries...")
def load_statistical_summary(data_dir: str):
    prepared = load_nhanes_data(data_dir)
    return prepared, summarize_diabetes_data(prepared.data)


try:
    model, metadata = load_model()
    model_available = True
    model_error = None

except Exception as exc:
    model_available = False
    model_error = str(exc)


# ============================================================
# APPLICATION HEADER
# ============================================================

st.title("StatFit")

st.subheader(
    "Experimental Classification of Existing, Self-Reported Diabetes Status"
)

st.info(
    "StatFit is an academic preventive-health analytics prototype. "
    "Its experimental diabetes model estimates existing, "
    "self-reported diabetes status, not future disease risk."
)

if not model_available:
    st.error(
        "The diabetes model failed to load or validate. "
        "The assessment interface remains available, but "
        "model predictions are disabled."
    )

    with st.expander("Technical details"):
        st.code(model_error)


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.title("StatFit")

page = st.sidebar.radio(
    "Navigation",
    [
        "Research & Model Evaluation",
        "Statistical Analysis",
        "Dataset & Methodology",
        "Health Assessment",
        "Risk Dashboard",
        "Recommendations",
        "About"
    ]
)


# ============================================================
# HEALTH ASSESSMENT
# ============================================================

if page == "Health Assessment":

    st.header("Health & Lifestyle Assessment")

    st.write(
        "This interactive health assessment is a prototype workflow. "
        "The primary research presentation focuses on NHANES cohort analysis "
        "and classification of existing reported diabetes status."
    )

    st.caption(
        "The current experimental diabetes model uses age, sex, "
        "BMI, waist circumference and blood pressure."
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Personal Information")

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=80,
            value=20
        )

        sex = st.selectbox(
            "Sex",
            ["Male", "Female"]
        )

        height = st.number_input(
            "Height (cm)",
            min_value=100.0,
            max_value=230.0,
            value=178.0
        )

        weight = st.number_input(
            "Weight (kg)",
            min_value=30.0,
            max_value=250.0,
            value=74.0
        )

        body_fat = st.number_input(
            "Body Fat (%)",
            min_value=2.0,
            max_value=60.0,
            value=15.0
        )

        waist = st.number_input(
            "Waist Circumference (cm)",
            min_value=40.0,
            max_value=200.0,
            value=85.0
        )

    with col2:

        st.subheader("Lifestyle & Clinical Measurements")

        calories = st.number_input(
            "Average Daily Calories",
            min_value=500,
            max_value=6000,
            value=2200
        )

        protein = st.number_input(
            "Protein (g/day)",
            min_value=0.0,
            max_value=400.0,
            value=100.0
        )

        activity = st.selectbox(
            "Average Activity Level",
            [
                "Sedentary",
                "Lightly Active",
                "Moderately Active",
                "Very Active"
            ]
        )

        sleep = st.number_input(
            "Average Sleep (hours)",
            min_value=0.0,
            max_value=16.0,
            value=7.0
        )

        st.caption(
            "Enter available readings. At least one systolic and one "
            "diastolic value are needed for the diabetes classifier."
        )

        systolic_readings = [
            st.number_input(
                f"Systolic reading {index} (mmHg)",
                min_value=60,
                max_value=250,
                value=None,
                placeholder="Optional",
                key=f"systolic_reading_{index}",
            )
            for index in range(1, 4)
        ]

        diastolic_readings = [
            st.number_input(
                f"Diastolic reading {index} (mmHg)",
                min_value=30,
                max_value=150,
                value=None,
                placeholder="Optional",
                key=f"diastolic_reading_{index}",
            )
            for index in range(1, 4)
        ]

    medical_history = st.multiselect(
        "Relevant Medical History",
        [
            "Diabetes",
            "Hypertension",
            "High Cholesterol",
            "Cardiovascular Disease",
            "None"
        ]
    )

    if st.button("Analyze Health Profile", type="primary"):

        for output_key in (
            "diabetes_status_probability",
            "diabetes_status_classification",
            "diabetes_status_threshold",
        ):
            st.session_state.pop(output_key, None)

        try:
            systolic_avg, diastolic_avg = average_blood_pressure_readings(
                systolic_readings,
                diastolic_readings,
            )
        except (TypeError, ValueError) as exc:
            systolic_avg, diastolic_avg = None, None
            st.error(f"Blood pressure readings are invalid: {exc}")

        height_m = height / 100
        bmi = weight / (height_m ** 2)

        st.session_state["bmi"] = bmi
        st.session_state["age"] = age
        st.session_state["sex"] = sex
        st.session_state["body_fat"] = body_fat
        st.session_state["waist"] = waist
        st.session_state["calories"] = calories
        st.session_state["protein"] = protein
        st.session_state["activity"] = activity
        st.session_state["sleep"] = sleep
        st.session_state["systolic"] = systolic_avg
        st.session_state["diastolic"] = diastolic_avg
        st.session_state["medical_history"] = medical_history

        st.success("Health profile processed successfully.")

        st.divider()

        st.subheader("Calculated Health Metrics")

        m1, m2, m3, m4 = st.columns(4)

        m1.metric("BMI", f"{bmi:.1f}")
        m2.metric("Body Fat", f"{body_fat:.1f}%")
        m3.metric("Calories", f"{calories} kcal")
        m4.metric("Sleep", f"{sleep:.1f} hrs")

        # ----------------------------------------------------
        # EXPERIMENTAL DIABETES STATUS MODEL
        # ----------------------------------------------------

        if model_available and systolic_avg is not None and diastolic_avg is not None:

            # Use the same pure mapping function exercised by tests.
            model_input = build_prediction_input(
                age=age,
                sex=sex,
                bmi=bmi,
                waist_cm=waist,
                systolic_mm_hg=systolic_avg,
                diastolic_mm_hg=diastolic_avg
            )

            expected_features = metadata["features"]

            # Prevent silent errors if the saved model's
            # expected feature order differs.
            if list(model_input.columns) != expected_features:

                st.error(
                    "The application's input features do not match "
                    "the saved model's expected features. "
                    "Predictions have been disabled."
                )

            else:

                positive_class_index = list(model.classes_).index(1)
                probability = float(
                    model.predict_proba(model_input)[0, positive_class_index]
                )

                threshold = float(metadata["threshold"])

                classification = int(
                    probability >= threshold
                )

                st.session_state[
                    "diabetes_status_probability"
                ] = probability

                st.session_state[
                    "diabetes_status_classification"
                ] = classification

                st.session_state[
                    "diabetes_status_threshold"
                ] = threshold

                st.divider()

                st.subheader(
                    "Experimental Diabetes Status Model"
                )

                st.metric(
                    "Model-estimated probability",
                    f"{probability:.1%}"
                )

                if classification == 1:

                    st.warning(
                        "The model classified this profile into "
                        "the positive class for reported diabetes "
                        "status. This is not a diagnosis."
                    )

                else:

                    st.info(
                        "The model classified this profile into "
                        "the negative class for reported diabetes "
                        "status. This does not rule out diabetes."
                    )

                st.caption(
                    f"Experimental classification threshold: "
                    f"{threshold:.2f}. The model was trained on "
                    f"NHANES 2021–2023 data and has not been "
                    f"clinically validated. Its output must not "
                    f"be interpreted as an individual's validated "
                    f"probability of having or developing diabetes."
                )

        elif model_available:

            st.warning(
                "No diabetes status prediction was generated. Enter at least "
                "one valid systolic and one valid diastolic reading. Missing "
                "readings are not treated as zero."
            )

        else:

            st.warning(
                "The diabetes status model is unavailable. "
                "Health metrics were calculated, but no model "
                "prediction was generated."
            )


# ============================================================
# RISK DASHBOARD
# ============================================================

elif page == "Research & Model Evaluation":

    st.header("Research & Model Evaluation")
    st.write(
        "Compare the original model used by the assessment workflow with the "
        "separate Stage 1 pipeline candidate."
    )
    st.info(
        "Both are development evaluations on previously examined test participants, "
        "not independent validation or an untouched final test. The candidate is not "
        "used by the app. These models classify existing self-reported doctor-diagnosed "
        "diabetes status; they do not estimate future disease incidence."
    )
    try:
        evaluation = load_json_report(str(EVALUATION_PATH))
        candidate_evaluation = load_json_report(str(STAGE1_CANDIDATE_EVALUATION_PATH))
        candidate_metadata = load_json_report(str(PIPELINE_METADATA_PATH))
        original_matrix = evaluation["confusion_matrix"]
        candidate_matrix = candidate_evaluation["confusion_matrix"]
        for name, matrix, report in (
            ("Original", original_matrix, evaluation),
            ("Candidate", candidate_matrix, candidate_evaluation),
        ):
            if len(matrix) != 2 or any(len(row) != 2 for row in matrix):
                raise ValueError(f"Saved {name.lower()} confusion matrix must be 2 by 2.")
            if sum(int(value) for row in matrix for value in row) != int(report["test_samples"]):
                raise ValueError(f"Saved {name.lower()} confusion matrix count does not match test_samples.")
        tn, fp, fn, tp = (int(value) for row in original_matrix for value in row)
        specificity = tn / (tn + fp) if tn + fp else None

        st.caption(
            f"Target: {evaluation.get('target', TARGET_DEFINITION)}. "
            f"Original app model threshold: {float(evaluation['threshold']):.2f}; "
            f"candidate threshold: {float(candidate_evaluation['threshold']):.2f}. "
            f"Participants per evaluation: {int(evaluation['test_samples']):,}."
        )

        comparison = build_evaluation_comparison(
            evaluation, candidate_evaluation, candidate_metadata
        )
        original_label, candidate_label = comparison.columns
        original_recall = float(comparison.loc["Recall", original_label])
        candidate_recall = float(comparison.loc["Recall", candidate_label])
        recall_change_pp = (candidate_recall - original_recall) * 100

        st.subheader("Development model comparison")
        finding_col, scope_col = st.columns([1, 2])
        finding_col.metric(
            "Candidate recall",
            f"{candidate_recall:.1%}",
            delta=f"{recall_change_pp:+.1f} percentage points",
        )
        scope_col.markdown(
            "**At the saved thresholds:** candidate recall is higher, while precision is lower. "
            "Recall measures the fraction of participants with reported diabetes status that the "
            "model classifies positive; precision measures the fraction of positive classifications "
            "that match reported-positive status."
        )
        st.bar_chart(comparison, y_label="Score (0–1)", stack=False)
        st.caption(
            "Precision is the share of positive classifications that match reported diabetes status; "
            "recall is the share of reported-positive participants detected. At the saved thresholds, "
            "the candidate has higher recall and lower precision, reflecting a precision–recall trade-off."
        )
        st.dataframe(comparison.round(3), width="stretch")

        st.subheader("Original app model metrics")
        metric_columns = st.columns(6)
        metric_values = [
            ("Precision", evaluation["precision"]),
            ("Recall", evaluation["recall"]),
            ("F1-score", evaluation["f1_score"]),
            ("ROC-AUC", evaluation["roc_auc"]),
            ("Specificity", specificity),
            ("Brier score", evaluation["brier_score"]),
        ]
        for column, (label, value) in zip(metric_columns, metric_values):
            column.metric(label, "Unavailable" if value is None else f"{float(value):.3f}")
        st.subheader("Confusion-count comparison")
        candidate_tn, candidate_fp, candidate_fn, candidate_tp = (
            int(value) for row in candidate_matrix for value in row
        )
        confusion_counts = pd.DataFrame(
            [
                {"Model": "Original app model (active)", "TN": tn, "FP": fp, "FN": fn, "TP": tp},
                {"Model": "Stage 1 candidate (not used by app)", "TN": candidate_tn, "FP": candidate_fp, "FN": candidate_fn, "TP": candidate_tp},
            ]
        )
        st.dataframe(confusion_counts, width="stretch", hide_index=True)
        st.caption(
            "TN: correctly classified no reported diabetes. TP: correctly classified reported diabetes. "
            "FP: classified positive when no diabetes was reported. FN: classified negative when diabetes was reported."
        )
        st.caption("Specificity is derived from each saved matrix: TN / (TN + FP).")

        with st.expander("Metric definitions"):
            st.dataframe(
                pd.DataFrame(
                    [
                        {"Metric": "Precision", "Meaning": "Of positive classifications, the fraction matching reported-positive status."},
                        {"Metric": "Recall", "Meaning": "Of reported-positive participants, the fraction classified positive."},
                        {"Metric": "F1-score", "Meaning": "Harmonic mean of precision and recall at the saved threshold."},
                        {"Metric": "ROC-AUC", "Meaning": "Ranking discrimination across thresholds; not a calibration measure."},
                    ]
                ),
                width="stretch",
                hide_index=True,
            )

        st.subheader("NHANES cohort and experiment")
        preparation_counts = candidate_metadata.get("preparation_counts", {})
        positive_count = preparation_counts.get("positive_target_rows")
        negative_count = preparation_counts.get("negative_target_rows")
        cohort_count = preparation_counts.get("adult_complete_case_rows")
        if all(value is not None for value in (positive_count, negative_count, cohort_count)):
            st.caption(
                f"Stage 1 pipeline adult complete-case cohort: {int(cohort_count):,} participants "
                f"({int(positive_count):,} reported positive; {int(negative_count):,} reported negative)."
            )
            st.bar_chart(
                pd.Series(
                    {"No reported diabetes": int(negative_count), "Reported diabetes": int(positive_count)},
                    name="Participants",
                ),
                y_label="Participants",
            )
        else:
            st.caption("Cohort class counts: unavailable in the saved pipeline metadata.")

        split_counts = candidate_metadata.get("split_counts")
        if isinstance(split_counts, dict) and all(name in split_counts for name in ("train", "validation", "test")):
            split_table = pd.DataFrame.from_dict(split_counts, orient="index")
            split_table.index.name = "Stage 1 split"
            st.dataframe(split_table, width="stretch")
            overlaps = candidate_metadata.get("participant_split_integrity")
            st.caption(
                f"Stage 1 split record: seed {candidate_metadata.get('split_seed', 'unavailable')}; "
                f"participant overlaps {overlaps if overlaps is not None else 'unavailable'}. "
                "The original app artifact's full training provenance is unavailable."
            )
        else:
            st.caption("Stage 1 split counts: unavailable in the saved pipeline metadata.")

        st.subheader("Research workflow")
        st.markdown(
            "1. Join local NHANES 2021–2023 demographics, body measures, diabetes questionnaire, and blood-pressure files by participant ID.\n"
            "2. Map DIQ010=1 to reported diabetes and DIQ010=2 to no reported diabetes; retain adults with complete values for the six documented features.\n"
            "3. Create the recorded stratified 60/20/20 Stage 1 split (seed 2026); compare Random Forest `min_samples_leaf=1` and `5`.\n"
            "4. Select configuration and threshold on validation data (maximum validation F1 over the recorded threshold grid); report test results as development results because participants were previously examined."
        )
        st.caption("Stage 1 model inputs: " + ", ".join(FEATURES))
        threshold_selection = candidate_metadata.get("threshold_selection", {})
        if threshold_selection:
            st.caption(
                "Validation-only threshold selection: "
                f"{threshold_selection.get('criterion', 'criterion unavailable')}; "
                f"tie-break: {threshold_selection.get('tie_break', 'unavailable')}; "
                f"threshold grid: {threshold_selection.get('threshold_grid', 'unavailable')}."
            )
        st.caption(
            "The active app continues to use the original saved Random Forest and its saved threshold. "
            "Candidate results are experimental; individualized future-disease-risk prediction remains future scope."
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.error(f"The saved evaluation report could not be displayed: {exc}")

elif page == "Statistical Analysis":

    st.header("Statistical Analysis")
    st.write(
        "Descriptive summaries use the adult, complete-case NHANES records "
        "prepared by the current diabetes pipeline."
    )
    st.info(
        "These summaries describe associations in the observed sample. They "
        "are not causal estimates, clinical conclusions, or future-risk predictions."
    )
    try:
        prepared, summary = load_statistical_summary(str(RAW_DATA_DIR))
        st.caption(
            f"Prepared sample: {summary['participant_count']:,} participants. "
            "The class outcome is existing self-reported doctor-diagnosed diabetes status."
        )
        st.subheader("Diabetes-status class distribution")
        st.bar_chart(summary["class_distribution"]["participants"], y_label="Participants")
        st.dataframe(summary["class_distribution"].round(2), width="stretch")

        unit_labels = {
            "RIDAGEYR": "Age (years)",
            "BMXBMI": "BMI (kg/m²)",
            "BMXWAIST": "Waist circumference (cm)",
            "systolic_avg": "Average systolic blood pressure (mmHg)",
            "diastolic_avg": "Average diastolic blood pressure (mmHg)",
        }
        st.subheader("Distributions by reported diabetes status")
        st.caption("Each status group is shown as a percentage within that group.")
        features = list(unit_labels)
        for start in range(0, len(features), 2):
            columns = st.columns(2)
            for column, feature in zip(columns, features[start:start + 2]):
                with column:
                    st.markdown(f"**{unit_labels[feature]}**")
                    st.bar_chart(
                        summary["distributions"][feature],
                        y_label="Participants within status (%)",
                    )

        st.subheader("Descriptive statistics by status")
        descriptive = summary["descriptive_statistics"].copy()
        descriptive["measure"] = descriptive["feature"].map(unit_labels)
        descriptive = descriptive[["status", "measure", "n", "mean", "median", "standard_deviation"]]
        descriptive = descriptive.rename(columns={"n": "N", "standard_deviation": "Standard deviation"})
        st.dataframe(descriptive.round(2), width="stretch", hide_index=True)
        st.caption(
            "N is the non-missing count for each measure. Standard deviation uses "
            "the sample formula (ddof=1). Group differences are descriptive only."
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.error(f"NHANES descriptive analysis could not be prepared: {exc}")

elif page == "Dataset & Methodology":

    st.header("Dataset & Methodology")
    st.write(
        "The current app model uses NHANES 2021–2023 component files available "
        "locally. The details below distinguish the active model artifact from "
        "the reproducibility pipeline experiment."
    )
    st.subheader("Dataset preparation")
    st.markdown(
        """
        - **Source components:** DEMO_L.xpt (demographics), BMX_L.xpt (body measures),
          DIQ_L.xpt (diabetes questionnaire), and BPXO_L.xpt (blood-pressure readings).
        - **Joins:** participant-level `SEQN` joins; demographics, body measures and
          diabetes questionnaire are inner-joined, then blood pressure is left-joined.
        - **Target:** `DIQ010=1` maps to reported diabetes (1); `DIQ010=2` maps to
          no reported diabetes (0). Other or missing responses are excluded.
        - **Eligibility:** adults age 18 or older with complete values for all six
          model features are retained.
        """
    )
    st.subheader("Model inputs")
    st.code("  →  ".join(FEATURES), language="text")
    st.caption(
        "The app computes BMI from entered height and weight, maps Male/Female to "
        "NHANES RIAGENDR values, and averages available systolic and diastolic readings."
    )
    st.subheader("Reproducibility pipeline")
    try:
        pipeline_metadata = load_json_report(str(PIPELINE_METADATA_PATH))
        split_counts = pipeline_metadata["split_counts"]
        threshold = pipeline_metadata["threshold"]
        threshold_method = pipeline_metadata["threshold_selection"]["criterion"]
        st.markdown(
            f"- Stratified train/validation/test split: **{pipeline_metadata['split_manifest']['proportions']}**, "
            f"seed **{pipeline_metadata['split_seed']}**.\n"
            f"- Split rows: **{split_counts['train']['rows']:,} / {split_counts['validation']['rows']:,} / {split_counts['test']['rows']:,}** "
            "(train / validation / test).\n"
            f"- Random Forest configurations compared: `min_samples_leaf={RF_CONFIGURATIONS[0]}` and "
            f"`min_samples_leaf={RF_CONFIGURATIONS[1]}`.\n"
            f"- Pipeline-selected threshold: **{float(threshold):.2f}**, chosen using {threshold_method}."
        )
        st.caption(
            "These split and selection details describe the Stage 1 reproducibility "
            "pipeline candidate. They are not established training provenance for "
            "the original model currently loaded by the app."
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        st.warning(f"Pipeline metadata is unavailable; only source-code methodology is shown. ({exc})")
    st.subheader("Active app artifact")
    st.write(
        f"The Health Assessment continues to load `{MODEL_PATH.name}` with threshold "
        f"{float(metadata['threshold']):.2f} when the original artifact is available. "
        "Its saved metadata does not include the full split and training-parameter provenance."
        if model_available else
        "The original app model is currently unavailable; model predictions remain disabled."
    )

elif page == "Risk Dashboard":

    st.header("StatFit Dashboard")
    st.caption(
        "This prototype dashboard summarizes submitted measurements and the experimental "
        "existing-status classifier. Individual future-disease-risk prediction remains future scope."
    )

    st.write(
        "Review the health metrics and experimental model output "
        "generated from the most recently submitted assessment."
    )

    if "bmi" not in st.session_state:

        st.info(
            "Complete the Health Assessment first to populate "
            "the dashboard."
        )

    else:

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "BMI",
            f"{st.session_state['bmi']:.1f}"
        )

        col2.metric(
            "Waist Circumference",
            f"{st.session_state['waist']:.1f} cm"
        )

        col3.metric(
            "Average Systolic Blood Pressure",
            (
                f"{st.session_state['systolic']:.1f} mmHg"
                if st.session_state["systolic"] is not None
                else "Not entered"
            )
        )

        st.subheader("Experimental Model Output")

        if "diabetes_status_probability" in st.session_state:

            st.metric(
                "Model-estimated probability of reported "
                "diabetes status",
                f"{st.session_state['diabetes_status_probability']:.1%}"
            )

            st.write(
                "This score concerns existing, self-reported "
                "diabetes status in the model's training data. "
                "It is not a validated future-risk estimate."
            )

        else:

            st.info(
                "No diabetes model output is available yet. "
                "Complete the Health Assessment to generate it."
            )

        st.subheader("Other Disease Models")

        st.write(
            "Hypertension, cardiovascular disease and other "
            "disease prediction models have not yet been "
            "implemented in this application."
        )


# ============================================================
# RECOMMENDATIONS
# ============================================================

elif page == "Recommendations":

    st.header("Personalized Health Recommendations")
    st.caption(
        "Individualized future-risk prediction and personalized recommendations remain prototype/future scope."
    )

    st.write(
        "This section will be expanded to provide evidence-based "
        "recommendations using validated methods and the "
        "individual's health and lifestyle information."
    )

    if "bmi" not in st.session_state:

        st.info(
            "Complete the Health Assessment to make your "
            "calculated health metrics available here."
        )

    else:

        bmi = st.session_state["bmi"]

        st.subheader("Calculated Health Metrics")

        st.write(f"**BMI:** {bmi:.1f}")

        if bmi < 18.5:

            st.write(
                "BMI is below the conventional adult reference "
                "range. Individual assessment is needed."
            )

        elif bmi < 25:

            st.write(
                "BMI is within the conventional adult reference "
                "range. BMI alone does not establish health status."
            )

        elif bmi < 30:

            st.write(
                "BMI is within the conventional overweight "
                "category. Interpret it alongside other measurements."
            )

        else:

            st.write(
                "BMI is within the conventional obesity category. "
                "BMI alone cannot determine an individual's "
                "health or disease risk."
            )

        st.subheader("Lifestyle")

        st.write(
            f"**Activity level:** "
            f"{st.session_state['activity']}"
        )

        st.write(
            f"**Sleep:** {st.session_state['sleep']:.1f} hours"
        )

        st.write(
            f"**Protein intake:** "
            f"{st.session_state['protein']:.1f} g/day"
        )

        st.info(
            "Personalized disease-specific recommendations are "
            "not yet clinically validated or implemented."
        )


# ============================================================
# ABOUT
# ============================================================

elif page == "About":

    st.header("About StatFit")

    st.write(
        """
        StatFit is a research prototype for NHANES data preparation,
        descriptive analysis, and experimental classification of existing,
        self-reported diabetes status. The individual assessment interface
        is a demonstration workflow; future-risk prediction remains future scope.

        Current workflow:

        1. User health and lifestyle data collection
        2. Basic health metric calculation
        3. Experimental diabetes status classification
        4. Dashboard display

        Planned future development:

        1. Evidence-based risk prediction
        2. Additional disease models
        3. Personalized lifestyle recommendations
        4. Model validation and uncertainty analysis
        5. Clinical decision-support research
        """
    )

    st.warning(
        "The current diabetes model estimates existing, "
        "self-reported diabetes status rather than future disease "
        "incidence. It has not been clinically validated and "
        "must not be used for diagnosis, treatment decisions "
        "or medical screening."
    )
