# Delayed Emergence GBM Streamlit Calculator

这是基于论文最终 GBM 模型设计的 Streamlit 网页计算器版本。它参考 AKI 文献的实现路线：输入模型变量，输出风险概率，并在模型和 SHAP 环境可用时显示个体化 SHAP waterfall/force plot。

## 已部署网址

- Streamlit 公网应用：https://delayed-emergence-gbm-calculator.streamlit.app/
- GitHub 仓库：https://github.com/xiaodongZhang88/delayed-emergence-gbm-calculator

## 目录结构

```text
gbm-streamlit-calculator/
├── streamlit_app.py
├── requirements.txt
├── models/
│   └── final_gbm_model.joblib 或 final_gbm_model.pkl
├── data/
│   └── background.csv
└── .streamlit/
    └── config.toml
```

## 需要提供的文件

必须提供最终训练好的 GBM 模型文件：

- `models/final_gbm_model.joblib`
- 或 `models/final_gbm_model.pkl`

模型最好支持：

```python
model.predict_proba(X)
```

其中 `X` 的列名建议为：

```text
age, gnri, be, wbc, rbc, alt, urea
```

如果需要稳定的 SHAP 图，建议额外提供训练集背景数据：

```text
data/background.csv
```

该 CSV 只需要包含这 7 个模型变量，列名与模型训练时一致。可从训练集抽取 100-200 行代表性样本，不建议上传完整敏感数据。

## 本地运行

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## 部署到 Streamlit Community Cloud

1. 新建 GitHub 仓库。
2. 上传本目录所有文件。
3. 把最终模型放到 `models/final_gbm_model.joblib` 或 `models/final_gbm_model.pkl`。
4. 如果要显示 SHAP 图，把背景数据放到 `data/background.csv`。
5. 打开 [Streamlit Community Cloud](https://streamlit.io/cloud)。
6. 选择 GitHub 仓库，入口文件填 `streamlit_app.py`。
7. 部署后会得到类似 `https://your-app.streamlit.app/` 的公网网址。

## 注意

当前版本不会伪造预测概率。没有最终模型文件时，网页只显示输入界面和模型接入提示。
