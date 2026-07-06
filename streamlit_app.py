from __future__ import annotations

import gzip
import io
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


APP_DIR = Path(__file__).resolve().parent
MODEL_PATHS = [
    APP_DIR / "models" / "final_gbm_model.json.gz",
    APP_DIR / "models" / "final_gbm_model.json",
    APP_DIR / "models" / "final_gbm_model.joblib",
    APP_DIR / "models" / "final_gbm_model.pkl",
    APP_DIR / "models" / "final_gbm_model.pickle",
]
BACKGROUND_PATH = APP_DIR / "data" / "background.csv"


@dataclass(frozen=True)
class FeatureSpec:
    key: str
    label: str
    description: str
    unit: str
    min_value: float
    max_value: float
    default: float
    step: float
    help_text: str


FEATURES = [
    FeatureSpec("age", "Age", "Patient age", "years", 18, 100, 62, 1, "Median in the study cohort: 62 years."),
    FeatureSpec(
        "gnri",
        "GNRI",
        "Geriatric Nutritional Risk Index",
        "score",
        60,
        130,
        102.90,
        0.1,
        "Median in the study cohort: 102.90.",
    ),
    FeatureSpec("be", "Base excess", "Base excess", "mmol/L", -20, 20, 0.81, 0.1, "Mean in the study cohort: 0.81 mmol/L."),
    FeatureSpec(
        "wbc",
        "WBC count",
        "White blood cell count",
        "10^9/L",
        0.5,
        50,
        5.50,
        0.01,
        "Median in the study cohort: 5.50 10^9/L.",
    ),
    FeatureSpec(
        "rbc",
        "RBC count",
        "Red blood cell count",
        "10^12/L",
        1,
        8,
        4.46,
        0.01,
        "Mean in the study cohort: 4.46 10^12/L.",
    ),
    FeatureSpec("alt", "ALT", "Alanine aminotransferase", "U/L", 1, 1000, 14, 1, "Median in the study cohort: 14 U/L."),
    FeatureSpec("urea", "Urea", "Urea", "mmol/L", 1, 50, 5.15, 0.01, "Median in the study cohort: 5.15 mmol/L."),
]

FEATURE_KEYS = [item.key for item in FEATURES]

FEATURE_ALIASES = {
    "age": {"age", "Age", "AGE"},
    "gnri": {"gnri", "GNRI", "Geriatric Nutritional Risk Index"},
    "be": {"be", "BE", "Base excess", "base_excess", "BaseExcess"},
    "wbc": {"wbc", "WBC", "WBC count", "wbc_count", "White blood cell count"},
    "rbc": {"rbc", "RBC", "RBC count", "rbc_count", "Red blood cell count"},
    "alt": {"alt", "ALT", "Alanine aminotransferase"},
    "urea": {"urea", "Urea", "BUN", "bun"},
}


class RJsonGBM:
    """Minimal predictor for JSON exports from R package gbm."""

    def __init__(self, payload: dict[str, Any]):
        if payload.get("format") != "r-gbm-json-v1":
            raise ValueError("Unsupported R GBM JSON format.")

        self.payload = payload
        self.initF = float(payload["initF"])
        self.var_names = [str(item) for item in payload["var_names"]]
        self.feature_names_in_ = np.array(self.var_names, dtype=object)
        self.n_features_in_ = len(self.var_names)
        self.classes_ = np.array([0, 1])
        self.trees = payload["trees"]
        self.n_trees = int(payload.get("n_trees", len(self.trees)))
        self.source = payload.get("source", "R gbm JSON")
        self.training_summary = payload.get("training_summary", {})

    def _as_matrix(self, x: Any) -> np.ndarray:
        if isinstance(x, pd.DataFrame):
            missing = [name for name in self.var_names if name not in x.columns]
            if missing:
                raise ValueError(f"Model input is missing required variables: {', '.join(missing)}")
            return x.loc[:, self.var_names].to_numpy(dtype=float)

        arr = np.asarray(x, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != self.n_features_in_:
            raise ValueError(f"Model requires {self.n_features_in_} variables, but received {arr.shape[1]}.")
        return arr

    @staticmethod
    def _sigmoid(score: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-score))

    def decision_function(self, x: Any) -> np.ndarray:
        rows = self._as_matrix(x)
        scores = np.full(rows.shape[0], self.initF, dtype=float)

        for row_index, row in enumerate(rows):
            score = self.initF
            for tree in self.trees:
                node = 0
                split_var = tree["split_var"]
                split_code = tree["split_code"]
                left = tree["left"]
                right = tree["right"]
                missing = tree["missing"]
                prediction = tree["prediction"]

                while int(split_var[node]) != -1:
                    value = row[int(split_var[node])]
                    if np.isnan(value):
                        node = int(missing[node])
                    elif value < float(split_code[node]):
                        node = int(left[node])
                    else:
                        node = int(right[node])
                score += float(prediction[node])
            scores[row_index] = score

        return scores

    def predict_proba(self, x: Any) -> np.ndarray:
        probability = self._sigmoid(self.decision_function(x))
        return np.column_stack([1.0 - probability, probability])

    def predict(self, x: Any) -> np.ndarray:
        return self.predict_proba(x)[:, 1]

    def metadata_summary(self) -> str:
        n = self.training_summary.get("n")
        positive = self.training_summary.get("outcome_positive")
        negative = self.training_summary.get("outcome_negative")
        if n is None:
            return self.source
        return f"{self.source}; training n={n}, DL1=1: {positive}, DL1=0: {negative}"


def page_style() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            max-width: 1220px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .hero {
            border: 1px solid #dbe3ea;
            border-radius: 8px;
            padding: 22px 24px;
            background: #ffffff;
            margin-bottom: 18px;
        }
        .eyebrow {
            color: #0f766e;
            font-size: 0.76rem;
            font-weight: 760;
            letter-spacing: 0;
            text-transform: uppercase;
            margin-bottom: 0.25rem;
        }
        .hero h1 {
            margin: 0 0 0.35rem 0;
            font-size: 2.15rem;
            line-height: 1.08;
        }
        .hero p {
            color: #5f6f82;
            margin-bottom: 0;
        }
        .status-ok, .status-warn, .status-bad {
            border-radius: 8px;
            padding: 12px 14px;
            font-weight: 700;
            margin-bottom: 14px;
        }
        .status-ok {
            color: #115e59;
            background: #e8f5f3;
            border: 1px solid rgba(15, 118, 110, 0.28);
        }
        .status-warn {
            color: #7a4b00;
            background: #fff7e6;
            border: 1px solid rgba(183, 121, 31, 0.28);
        }
        .status-bad {
            color: #b42318;
            background: #fff1f0;
            border: 1px solid rgba(180, 35, 24, 0.28);
        }
        .metric-card {
            border: 1px solid #dbe3ea;
            border-radius: 8px;
            padding: 18px;
            background: #ffffff;
        }
        .small-muted {
            color: #64748b;
            font-size: 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def optional_imports() -> dict[str, Any]:
    modules: dict[str, Any] = {}
    for name in ["joblib", "shap", "matplotlib.pyplot"]:
        try:
            if name == "matplotlib.pyplot":
                import matplotlib.pyplot as plt

                modules["plt"] = plt
            elif name == "joblib":
                import joblib

                modules["joblib"] = joblib
            elif name == "shap":
                import shap

                modules["shap"] = shap
        except Exception:
            modules[name] = None
    return modules


def load_model_from_disk() -> tuple[Any | None, str | None]:
    for path in MODEL_PATHS:
        if path.exists():
            return load_model_bytes(path.read_bytes(), path.name), str(path.relative_to(APP_DIR))
    return None, None


def load_model_bytes(raw: bytes, filename: str) -> Any:
    lowered = filename.lower()
    if lowered.endswith(".json.gz"):
        payload = json.loads(gzip.decompress(raw).decode("utf-8"))
        return RJsonGBM(payload)
    if lowered.endswith(".json"):
        payload = json.loads(raw.decode("utf-8"))
        return RJsonGBM(payload)

    suffix = Path(filename).suffix.lower()
    modules = optional_imports()
    if suffix == ".joblib":
        joblib = modules.get("joblib")
        if joblib is None:
            raise RuntimeError("joblib is not available in the current environment, so the .joblib model cannot be loaded.")
        return joblib.load(io.BytesIO(raw))
    if suffix in {".pkl", ".pickle"}:
        return pickle.loads(raw)
    raise RuntimeError("Only .json.gz, .json, .joblib, .pkl, and .pickle model files are supported.")


@st.cache_data(show_spinner=False)
def load_background() -> pd.DataFrame | None:
    if not BACKGROUND_PATH.exists():
        return None
    data = pd.read_csv(BACKGROUND_PATH)
    return data


def model_feature_names(model: Any) -> list[str]:
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        return [str(item) for item in names]
    booster = getattr(model, "get_booster", lambda: None)()
    if booster is not None and getattr(booster, "feature_names", None):
        return [str(item) for item in booster.feature_names]
    return FEATURE_KEYS


def canonical_key(name: str) -> str | None:
    normalized = name.strip()
    compressed = "".join(ch for ch in normalized.lower() if ch.isalnum())
    for key, aliases in FEATURE_ALIASES.items():
        for alias in aliases:
            if normalized == alias:
                return key
            if compressed == "".join(ch for ch in alias.lower() if ch.isalnum()):
                return key
    return None


def make_feature_frame(values: dict[str, float], model: Any | None = None) -> pd.DataFrame:
    columns = model_feature_names(model) if model is not None else FEATURE_KEYS
    row: dict[str, float] = {}
    unresolved: list[str] = []
    for column in columns:
        key = canonical_key(column)
        if key is None or key not in values:
            unresolved.append(column)
            continue
        row[column] = values[key]

    if unresolved:
        raise RuntimeError(
            "The model feature names cannot be matched to the web inputs: "
            + ", ".join(unresolved)
            + ". Rename the model variables to age, gnri, be, wbc, rbc, alt, urea, or add aliases in FEATURE_ALIASES."
        )
    return pd.DataFrame([row], columns=columns)


def predict_probability(model: Any, x: pd.DataFrame) -> float:
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(x))
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return float(proba[0, 1])
        if proba.size == 1:
            return float(proba.ravel()[0])

    if hasattr(model, "decision_function"):
        score = float(np.asarray(model.decision_function(x)).ravel()[0])
        return float(1.0 / (1.0 + np.exp(-score)))

    if hasattr(model, "predict"):
        pred = float(np.asarray(model.predict(x)).ravel()[0])
        if 0 <= pred <= 1:
            return pred

    raise RuntimeError("Unable to obtain a probability from the model. Please provide a binary GBM model that supports predict_proba.")


def classify_risk(probability: float, medium: float, high: float) -> tuple[str, str]:
    if probability >= high:
        return "High risk", "#b42318"
    if probability >= medium:
        return "Intermediate risk", "#b7791f"
    return "Low risk", "#0f766e"


def make_shap_explanation(model: Any, x: pd.DataFrame):
    modules = optional_imports()
    shap = modules.get("shap")
    if shap is None:
        raise RuntimeError("shap is not available in the current environment, so the SHAP plot cannot be generated.")

    background = load_background()
    if background is not None:
        background = background.reindex(columns=x.columns)
        background = background.dropna(how="all")
        if len(background) > 200:
            background = background.sample(200, random_state=2026)

    if background is None or background.empty:
        background = x.copy()

    def positive_probability(data: Any) -> np.ndarray:
        if isinstance(data, pd.DataFrame):
            frame = data.reindex(columns=x.columns)
        else:
            frame = pd.DataFrame(np.asarray(data), columns=x.columns)
        if hasattr(model, "predict_proba"):
            return np.asarray(model.predict_proba(frame))[:, 1]
        return np.asarray(model.predict(frame)).ravel()

    try:
        explainer = shap.Explainer(model, background)
        shap_values = explainer(x)
    except Exception:
        explainer = shap.Explainer(positive_probability, background)
        shap_values = explainer(x)
    return normalize_shap_explanation(shap_values, x)


def normalize_shap_explanation(shap_values: Any, x: pd.DataFrame):
    modules = optional_imports()
    shap = modules.get("shap")
    values = np.asarray(shap_values.values)
    base_values = np.asarray(shap_values.base_values)

    if values.ndim == 3:
        values = values[:, :, -1]
        if base_values.ndim == 2:
            base_values = base_values[:, -1]

    return shap.Explanation(
        values=values[0],
        base_values=float(np.ravel(base_values)[0]),
        data=x.iloc[0].to_numpy(dtype=float),
        feature_names=list(x.columns),
    )


def render_shap(model: Any, x: pd.DataFrame) -> None:
    modules = optional_imports()
    shap = modules.get("shap")
    plt = modules.get("plt")
    if shap is None or plt is None:
        st.warning("shap or matplotlib is not available. The probability can be calculated, but SHAP plots cannot be displayed.")
        return

    try:
        explanation = make_shap_explanation(model, x)
    except Exception as exc:
        st.warning(f"SHAP plot generation failed: {exc}")
        return

    tab1, tab2 = st.tabs(["Waterfall plot", "Force plot"])
    with tab1:
        fig = plt.figure(figsize=(8.5, 4.8))
        shap.plots.waterfall(explanation, max_display=8, show=False)
        st.pyplot(fig, clear_figure=True)

    with tab2:
        try:
            force = shap.force_plot(
                explanation.base_values,
                explanation.values,
                pd.Series(explanation.data, index=explanation.feature_names),
                matplotlib=False,
            )
            components.html(f"{shap.getjs()}{force.html()}", height=230, scrolling=True)
        except Exception as exc:
            st.info(f"The force plot could not be rendered. The waterfall plot provides the same patient-level SHAP explanation. Error: {exc}")


def render_input_form() -> tuple[dict[str, float], float, float, bool]:
    st.subheader("Preoperative Variables")
    left, right = st.columns(2)
    values: dict[str, float] = {}
    for index, spec in enumerate(FEATURES):
        container = left if index % 2 == 0 else right
        with container:
            values[spec.key] = st.number_input(
                spec.label,
                min_value=float(spec.min_value),
                max_value=float(spec.max_value),
                value=float(spec.default),
                step=float(spec.step),
                help=f"{spec.help_text} Unit: {spec.unit}",
                format="%.2f" if spec.step < 1 else "%.0f",
            )
            st.caption(f"Unit: {spec.unit}")

    st.divider()
    st.subheader("Risk Stratification Thresholds")
    c1, c2, c3 = st.columns([1, 1, 1.2])
    with c1:
        medium = st.slider("Intermediate-risk threshold", 1, 80, 10, 1) / 100
    with c2:
        high = st.slider("High-risk threshold", 2, 95, 30, 1) / 100
    with c3:
        st.caption("Thresholds affect only the low/intermediate/high risk label and do not change the predicted probability.")

    if medium >= high:
        st.error("Risk thresholds must satisfy: intermediate-risk threshold < high-risk threshold.")
        can_predict = False
    else:
        can_predict = True
    return values, medium, high, can_predict


def model_status_box(model: Any | None, source: str | None) -> None:
    if model is None:
        st.markdown(
            """
            <div class="status-bad">
            Final GBM model was not detected. Place the model at
            <code>models/final_gbm_model.json.gz</code>,
            <code>models/final_gbm_model.joblib</code>, or
            <code>models/final_gbm_model.pkl</code>, then rerun the app.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    details = ""
    if hasattr(model, "metadata_summary"):
        details = f"<br><span style='font-weight: 500;'>{model.metadata_summary()}</span>"

    st.markdown(
        f"""
        <div class="status-ok">
        Loaded model: <code>{source or "uploaded model"}</code>{details}
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_model_upload() -> tuple[Any | None, str | None]:
    st.sidebar.header("Model Input")
    st.sidebar.caption("For deployment, place the model file in the models/ directory. You may upload a trusted model here for temporary testing.")
    uploaded = st.sidebar.file_uploader("Upload a trusted model file", type=["gz", "json", "joblib", "pkl", "pickle"])
    if uploaded is None:
        return load_model_from_disk()
    try:
        return load_model_bytes(uploaded.getvalue(), uploaded.name), uploaded.name
    except Exception as exc:
        st.sidebar.error(f"Model loading failed: {exc}")
        return None, None


def render_model_contract() -> None:
    with st.expander("Model File Requirements", expanded=False):
        st.markdown(
            """
            - The model should be a binary GBM/tree model and support `predict_proba(X)`.
            - R `gbm` models can be exported as `models/final_gbm_model.json.gz` for deployment.
            - Recommended input column names: `age`, `gnri`, `be`, `wbc`, `rbc`, `alt`, `urea`.
            - For stable SHAP plots, provide background data at `data/background.csv` with the same columns as the model.
            - This deployment uses the R GBM JSON model reconstructed and exported from `model.rds`; raw patient-level data are not uploaded.
            """
        )


def main() -> None:
    st.set_page_config(
        page_title="Delayed Emergence GBM Calculator",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    page_style()

    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">Gastrointestinal Cancer Surgery</div>
          <h1>Delayed Emergence GBM Risk Calculator</h1>
          <p>A seven-variable preoperative risk prediction tool for delayed emergence with individualized SHAP explanations.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    model, model_source = sidebar_model_upload()
    model_status_box(model, model_source)

    input_col, output_col = st.columns([1.05, 0.95], gap="large")
    with input_col:
        values, medium, high, can_predict = render_input_form()

    with output_col:
        st.subheader("Prediction Results")
        st.markdown(
            """
            <div class="status-warn">
            This tool is intended for manuscript model presentation and research validation only.
            Prospective external validation and local calibration are required before clinical use.
            </div>
            """,
            unsafe_allow_html=True,
        )

        predict_clicked = st.button("Calculate Risk and Generate SHAP Plot", type="primary", disabled=not can_predict)
        if not predict_clicked:
            st.info("Confirm the input values and click Calculate.")
            render_model_contract()
            return

        if model is None:
            st.error("The final GBM model has not been loaded, so a real prediction probability cannot be produced.")
            render_model_contract()
            return

        try:
            x = make_feature_frame(values, model)
            probability = predict_probability(model, x)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            render_model_contract()
            return

        probability = float(np.clip(probability, 0, 1))
        label, color = classify_risk(probability, medium, high)
        st.markdown(
            f"""
            <div class="metric-card">
              <div class="small-muted">Predicted probability</div>
              <div style="font-size: 3rem; font-weight: 820; color: {color}; line-height: 1.05;">
                {probability * 100:.1f}%
              </div>
              <div style="font-size: 1.2rem; font-weight: 760; color: {color};">{label}</div>
              <div class="small-muted">Intermediate risk >= {medium * 100:.0f}%, high risk >= {high * 100:.0f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Variable": spec.label,
                        "Description": spec.description,
                        "Value": values[spec.key],
                        "Unit": spec.unit,
                    }
                    for spec in FEATURES
                ]
            ),
            width="stretch",
            hide_index=True,
        )

        st.subheader("Individualized SHAP Explanation")
        with st.spinner("Generating individualized SHAP explanation..."):
            render_shap(model, x)


if __name__ == "__main__":
    main()
