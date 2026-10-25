# Data

This project uses the public **Telco Customer Churn** dataset:

- **Source:** Kaggle — <https://www.kaggle.com/datasets/blastchar/telco-customer-churn> (uploaded by BlastChar)
- **Origin:** IBM sample data set for customer retention analysis.
- **Size:** 7,043 customers × 21 columns (1 identifier, 19 features, 1 target `Churn`).

The raw CSV is **not redistributed** in this repository; all credit for the data goes to its original
publishers. To run the project, download it from Kaggle and place it here as:

```text
data/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

Optionally, with the [Kaggle CLI](https://github.com/Kaggle/kaggle-api) configured:

```powershell
kaggle datasets download -d blastchar/telco-customer-churn -p data --unzip
```

The file is treated as read-only: all cleaning happens in memory in `src/data_preprocessing.py`.
