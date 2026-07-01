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
│   └── final_gbm_model.json.gz
├── data/
│   └── background.csv
├── scripts/
│   └── export_rds_to_json.R
└── .streamlit/
    └── config.toml
```

## 当前模型

当前部署已接入 `/Users/zhangxiaodong/Downloads/model.rds` 导出的 GBM 模型：

- 模型文件：`models/final_gbm_model.json.gz`
- 模型来源：从 R `mlr3`/`gbm` 结果重建全数据 GBM 后导出
- 变量：`ALT`, `BE`, `GNRI`, `RBC`, `Urea`, `WBC`, `age`
- 结局：`DL1`，阳性类别为 `1`
- 背景数据：`data/background.csv` 仅包含 7 个变量的中位数，不包含原始病例行

网页端也兼容 Python 模型文件：

```text
models/final_gbm_model.joblib
models/final_gbm_model.pkl
models/final_gbm_model.pickle
```

## 重新导出模型

如果 RDS 文件更新，可以重新导出：

```bash
R_LIBS_USER=/path/to/Rlib Rscript scripts/export_rds_to_json.R /path/to/model.rds .
```

导出脚本会生成 `models/final_gbm_model.json.gz` 和 `data/background.csv`。请不要把完整原始病例数据上传到公开仓库。

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
3. 确认 `models/final_gbm_model.json.gz` 已存在。
4. 如果要显示 SHAP 图，确认 `data/background.csv` 已存在。
5. 打开 [Streamlit Community Cloud](https://streamlit.io/cloud)。
6. 选择 GitHub 仓库，入口文件填 `streamlit_app.py`。
7. 部署后会得到类似 `https://your-app.streamlit.app/` 的公网网址。

## 注意

当前版本不会伪造预测概率。若删除模型文件，网页会只显示输入界面和模型接入提示。
