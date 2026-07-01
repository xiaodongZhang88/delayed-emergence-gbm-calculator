# Model Placeholder

把最终训练好的 GBM 模型放在本目录，并命名为：

- `final_gbm_model.joblib`
- 或 `final_gbm_model.pkl`
- 或 `final_gbm_model.pickle`

推荐模型输入列名：

```text
age, gnri, be, wbc, rbc, alt, urea
```

模型应支持 `predict_proba(X)`，以便输出 delayed emergence 的预测概率。
