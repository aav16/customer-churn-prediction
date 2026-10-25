# Data

This project uses the **Telco Customer Churn** dataset (IBM sample data, distributed on Kaggle):
<https://www.kaggle.com/datasets/blastchar/telco-customer-churn>

The raw CSV is **not** committed to the repository. To run the project, download it and place it here as:

```text
data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

- 7,043 customers, 21 columns (1 identifier, 19 features, 1 target `Churn`).
- The file is treated as read-only: all cleaning happens in memory inside `src/data_preprocessing.py`.
