# Model File

当前部署使用：

- `final_gbm_model.json.gz`

该文件由 `scripts/export_rds_to_json.R` 从 R `model.rds` 导出，包含 GBM 树结构和必要元数据，不包含 688 行原始病例数据。

网页也兼容以下 Python 模型文件名：

- `final_gbm_model.joblib`
- 或 `final_gbm_model.pkl`
- 或 `final_gbm_model.pickle`

当前 R GBM 模型输入列名：

```text
ALT, BE, GNRI, RBC, Urea, WBC, age
```

Python 模型应支持 `predict_proba(X)`，以便输出 delayed emergence 的预测概率。
