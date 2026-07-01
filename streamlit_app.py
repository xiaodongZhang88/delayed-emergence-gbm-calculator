from __future__ import annotations

import io
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
    APP_DIR / "models" / "final_gbm_model.joblib",
    APP_DIR / "models" / "final_gbm_model.pkl",
    APP_DIR / "models" / "final_gbm_model.pickle",
]
BACKGROUND_PATH = APP_DIR / "data" / "background.csv"


@dataclass(frozen=True)
class FeatureSpec:
    key: str
    label: str
    zh: str
    unit: str
    min_value: float
    max_value: float
    default: float
    step: float
    help_text: str


FEATURES = [
    FeatureSpec("age", "Age", "年龄", "years", 18, 100, 62, 1, "论文总体队列中位数 62 岁。"),
    FeatureSpec("gnri", "GNRI", "老年营养风险指数", "score", 60, 130, 102.90, 0.1, "论文总体队列中位数 102.90。"),
    FeatureSpec("be", "Base excess", "碱剩余", "mmol/L", -20, 20, 0.81, 0.1, "论文总体队列均值 0.81。"),
    FeatureSpec("wbc", "WBC count", "白细胞计数", "×10⁹/L", 0.5, 50, 5.50, 0.01, "论文总体队列中位数 5.50。"),
    FeatureSpec("rbc", "RBC count", "红细胞计数", "×10¹²/L", 1, 8, 4.46, 0.01, "论文总体队列均值 4.46。"),
    FeatureSpec("alt", "ALT", "丙氨酸氨基转移酶 / 谷丙转氨酶", "U/L", 1, 1000, 14, 1, "论文总体队列中位数 14。"),
    FeatureSpec("urea", "Urea", "尿素", "mmol/L", 1, 50, 5.15, 0.01, "论文总体队列中位数 5.15。"),
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


@st.cache_resource(show_spinner=False)
def load_model_from_disk() -> tuple[Any | None, str | None]:
    for path in MODEL_PATHS:
        if path.exists():
            return load_model_bytes(path.read_bytes(), path.name), str(path.relative_to(APP_DIR))
    return None, None


def load_model_bytes(raw: bytes, filename: str) -> Any:
    suffix = Path(filename).suffix.lower()
    modules = optional_imports()
    if suffix == ".joblib":
        joblib = modules.get("joblib")
        if joblib is None:
            raise RuntimeError("当前环境缺少 joblib，无法读取 .joblib 模型。")
        return joblib.load(io.BytesIO(raw))
    if suffix in {".pkl", ".pickle"}:
        return pickle.loads(raw)
    raise RuntimeError("仅支持 .joblib、.pkl 或 .pickle 模型文件。")


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
            "模型需要的变量名无法和网页输入匹配："
            + ", ".join(unresolved)
            + "。请把模型变量名改为 age, gnri, be, wbc, rbc, alt, urea，或在 FEATURE_ALIASES 中添加别名。"
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

    raise RuntimeError("无法从模型获得概率。请提供支持 predict_proba 的二分类 GBM 模型。")


def classify_risk(probability: float, medium: float, high: float) -> tuple[str, str]:
    if probability >= high:
        return "高风险", "#b42318"
    if probability >= medium:
        return "中风险", "#b7791f"
    return "低风险", "#0f766e"


def make_shap_explanation(model: Any, x: pd.DataFrame):
    modules = optional_imports()
    shap = modules.get("shap")
    if shap is None:
        raise RuntimeError("当前环境缺少 shap，无法生成 SHAP 图。")

    background = load_background()
    if background is not None:
        background = background.reindex(columns=x.columns)
        background = background.dropna(how="all")
        if len(background) > 200:
            background = background.sample(200, random_state=2026)

    if background is not None and not background.empty:
        explainer = shap.Explainer(model, background)
    else:
        explainer = shap.Explainer(model)
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
        st.warning("当前环境缺少 shap 或 matplotlib，模型概率可计算，但不能显示 SHAP 图。")
        return

    try:
        explanation = make_shap_explanation(model, x)
    except Exception as exc:
        st.warning(f"SHAP 图生成失败：{exc}")
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
            st.info(f"force plot 暂时无法渲染，waterfall plot 已提供同一患者层面的 SHAP 解释。错误：{exc}")


def render_input_form() -> tuple[dict[str, float], float, float, bool]:
    st.subheader("患者术前指标")
    left, right = st.columns(2)
    values: dict[str, float] = {}
    for index, spec in enumerate(FEATURES):
        container = left if index % 2 == 0 else right
        with container:
            values[spec.key] = st.number_input(
                f"{spec.label} ({spec.zh})",
                min_value=float(spec.min_value),
                max_value=float(spec.max_value),
                value=float(spec.default),
                step=float(spec.step),
                help=f"{spec.help_text} 单位：{spec.unit}",
                format="%.2f" if spec.step < 1 else "%.0f",
            )
            st.caption(f"单位：{spec.unit}")

    st.divider()
    st.subheader("风险分层阈值")
    c1, c2, c3 = st.columns([1, 1, 1.2])
    with c1:
        medium = st.slider("中危起点", 1, 80, 10, 1) / 100
    with c2:
        high = st.slider("高危起点", 2, 95, 30, 1) / 100
    with c3:
        st.caption("阈值只影响低/中/高风险标签，不改变模型预测概率。")

    if medium >= high:
        st.error("风险阈值需满足：中危起点 < 高危起点。")
        can_predict = False
    else:
        can_predict = True
    return values, medium, high, can_predict


def model_status_box(model: Any | None, source: str | None) -> None:
    if model is None:
        st.markdown(
            """
            <div class="status-bad">
            未检测到最终 GBM 模型。请把模型放到 <code>models/final_gbm_model.joblib</code>
            或 <code>models/final_gbm_model.pkl</code>，然后重新运行。
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f"""
        <div class="status-ok">
        已加载模型：<code>{source or "uploaded model"}</code>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sidebar_model_upload() -> tuple[Any | None, str | None]:
    st.sidebar.header("模型接入")
    st.sidebar.caption("正式部署时建议把模型文件放入 models/ 目录。临时测试可在这里上传。")
    uploaded = st.sidebar.file_uploader("上传可信模型文件", type=["joblib", "pkl", "pickle"])
    if uploaded is None:
        return load_model_from_disk()
    try:
        return load_model_bytes(uploaded.getvalue(), uploaded.name), uploaded.name
    except Exception as exc:
        st.sidebar.error(f"模型读取失败：{exc}")
        return None, None


def render_model_contract() -> None:
    with st.expander("模型文件要求", expanded=False):
        st.markdown(
            """
            - 模型应为二分类 GBM/树模型，并支持 `predict_proba(X)`。
            - 输入变量建议使用这些列名：`age`, `gnri`, `be`, `wbc`, `rbc`, `alt`, `urea`。
            - 若要显示稳定的 SHAP 图，建议提供训练集背景数据：`data/background.csv`，列名与模型一致。
            - 如果模型来自 R 的 `gbm` 包，不能被 Python 直接读取；需要导出为 Python 可读取格式，或用 Python 重新训练/封装。
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
          <p>基于术前 7 变量的延迟苏醒风险预测与 SHAP 个体化解释工具。</p>
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
        st.subheader("预测结果")
        st.markdown(
            """
            <div class="status-warn">
            当前工具仅用于论文模型展示和研究验证。正式临床使用前需要前瞻性外部验证和本地校准。
            </div>
            """,
            unsafe_allow_html=True,
        )

        predict_clicked = st.button("计算风险并生成 SHAP 图", type="primary", disabled=not can_predict)
        if not predict_clicked:
            st.info("请确认输入值后点击计算。")
            render_model_contract()
            return

        if model is None:
            st.error("尚未接入最终 GBM 模型，不能输出真实预测概率。")
            render_model_contract()
            return

        try:
            x = make_feature_frame(values, model)
            probability = predict_probability(model, x)
        except Exception as exc:
            st.error(f"预测失败：{exc}")
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
              <div class="small-muted">中危 ≥ {medium * 100:.0f}%，高危 ≥ {high * 100:.0f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Variable": spec.label,
                        "中文": spec.zh,
                        "Value": values[spec.key],
                        "Unit": spec.unit,
                    }
                    for spec in FEATURES
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("个体化 SHAP 解释")
        render_shap(model, x)


if __name__ == "__main__":
    main()
