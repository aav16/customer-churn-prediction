# Methodology

This document explains the technical decisions behind the churn model and why each one was taken.
All numbers come from the pipeline run recorded in [`reports/model_results.md`](../reports/model_results.md).

## 1. Data and cleaning

| Issue found in EDA | Decision | Leakage risk |
|---|---|---|
| `TotalCharges` stored as text; 11 blank values | Parse to float; blanks belong to the 11 customers with `tenure == 0` (never billed) → set to `0` | None — deterministic rule, no statistics learned from data |
| `Churn` as `"Yes"/"No"` | Map to `1/0`; any other value raises an error | None |
| `customerID` | Kept for traceability, **never** used as a feature | — |
| `No internet service` / `No phone service` levels | Kept as their own category | — |
| Outliers (1.5×IQR rule) | None found; no treatment | — |
| 22 rows identical except the ID | Kept: plausible distinct customers with simple profiles | — |

Cleaning lives in `src/data_preprocessing.py::clean_data` and never modifies the CSV on disk.

## 2. Validation design

```text
7,043 customers
 ├── 80% train (5,634) ── 5-fold stratified CV ──► model, imbalance strategy & hyper-parameters
 │                     └─ out-of-fold predictions ─► classification threshold
 └── 20% test  (1,409) ──────────────────────────► final evaluation only (used once)
```

- The split is stratified on `Churn` (26.5% in both partitions), `random_state=42`.
- Everything that learns parameters (median imputation, standard scaling, one-hot categories,
  SMOTE, the model) sits inside one `imblearn.pipeline.Pipeline`. During cross-validation it is
  re-fitted on each training fold only, so validation folds never leak into preprocessing.
- The test set is reconstructed deterministically by later steps and checked against the customer
  IDs stored in `models/model_metadata.json`; a mismatch aborts the run.

## 3. Preprocessing pipeline

```text
ColumnTransformer
 ├── num: tenure, MonthlyCharges, TotalCharges, SeniorCitizen → SimpleImputer(median) → StandardScaler
 └── cat: 15 categorical columns                              → SimpleImputer(most_frequent) → OneHotEncoder(handle_unknown="ignore")
[SMOTE]      (only in the SMOTE strategy, applied on fit only)
Classifier   (LogisticRegression or XGBClassifier)
```

The imputers are a safety net for unexpected nulls in new data (the clean training data has none);
`handle_unknown="ignore"` makes scoring robust to unseen categories. Scaling helps the logistic
regression and is harmless for XGBoost; using one preprocessor for both keeps the comparison fair.

## 4. Class imbalance

Churn is 26.5% (ratio 2.77 : 1). A model predicting "no churn" for everyone scores 73.5% accuracy
and finds zero churners, so accuracy is reported only as a reference. The selection metric is
**PR-AUC (average precision)**, which focuses on how well the minority class is ranked.

Three strategies were compared inside cross-validation, for both models:

| Strategy | Logistic Regression | XGBoost |
|---|---|---|
| None | plain fit | plain fit |
| Class weighting | `class_weight="balanced"` | `scale_pos_weight = neg/pos = 2.769` |
| SMOTE | `SMOTE` after preprocessing, inside the CV pipeline | same |

Results (5-fold CV on training data; precision/recall/FP at threshold 0.5 from out-of-fold predictions):

| Model | Strategy | CV PR-AUC | Recall@0.5 | Precision@0.5 | False positives@0.5 |
|---|---|---|---|---|---|
| Logistic Regression | none | 0.662 | 0.543 | 0.653 | 432 |
| Logistic Regression | weighted | 0.660 | 0.801 | 0.517 | 1,120 |
| Logistic Regression | SMOTE | 0.659 | 0.777 | 0.532 | 1,021 |
| XGBoost | none | **0.672** | 0.520 | 0.667 | 388 |
| XGBoost | weighted | 0.671 | 0.817 | 0.518 | 1,139 |
| XGBoost | SMOTE | 0.665 | 0.664 | 0.595 | 677 |

**Interpretation.** Re-weighting and SMOTE barely change the *ranking* quality (PR-AUC within
0.007 of each other). What they mainly do is shift predicted probabilities upwards, which at a fixed 0.5
threshold trades precision for recall (false positives grow 2.6–2.9×). The same trade-off can
be obtained more transparently by **lowering the threshold** on the unweighted model, which also keeps
its probabilities calibrated (Brier 0.133 vs 0.162 for the weighted XGBoost). SMOTE has an extra
caveat: it interpolates one-hot columns into fractional values that no real customer can have.
The unweighted strategy therefore won on both PR-AUC and calibration.

## 5. Models and hyper-parameter search

Grid search with `GridSearchCV` (5-fold stratified CV, refit on PR-AUC), CPU only, fixed seeds:

| Model | Grid | Best (unweighted) |
|---|---|---|
| Logistic Regression (L2, lbfgs) | `C ∈ {0.01, 0.1, 1, 10}` | `C = 1.0` |
| XGBoost (`hist`, subsample 0.8, colsample 0.8) | `n_estimators ∈ {100, 200, 400}`, `max_depth ∈ {2, 3, 5}`, `learning_rate ∈ {0.03, 0.1}`, `min_child_weight ∈ {1, 5}` | `n_estimators=200, max_depth=3, learning_rate=0.03, min_child_weight=1` |

The XGBoost grid was widened once after the first run because the optimum sat on the edge of the
original grid; the widened search kept a shallow model (`max_depth=3`), which is consistent with
a mostly additive signal.

**Model selection rule:** highest mean CV PR-AUC → **XGBoost, no re-weighting (0.672 ± 0.023)**.
The margin over logistic regression (0.662 ± 0.019) is small, within one standard deviation, so the
logistic regression remains a credible, more interpretable alternative.

## 6. Threshold selection

The threshold is chosen on **out-of-fold predictions of the training set** (never on test), by
maximising F1 over a grid from 0.05 to 0.95:

- Selected: **0.33** (max F1).
- Considered: max F2 (recall weighted ×2) gives 0.19, but the F2 curve is almost flat between 0.14
  and 0.25 and flags ~49% of customers at a precision of ~0.47 — not a realistic campaign size.
  It is kept in the reports as a recall-oriented alternative.

Risk levels used in the predictions file are aligned with the threshold:

| Level | Rule |
|---|---|
| Low | probability < 0.33 (not flagged) |
| Medium | 0.33 ≤ probability < 0.60 |
| High | probability ≥ 0.60 |

## 7. Evaluation

Final metrics on the held-out test set (1,409 customers, 374 churners), XGBoost at threshold 0.33:
ROC-AUC 0.848, PR-AUC 0.663, recall 0.749, precision 0.557, F1 0.639, accuracy 0.775, Brier 0.135.
The calibration curve and the risk-level table (mean predicted probability ≈ observed churn rate in
each level) indicate that the probabilities can be read as approximate churn likelihoods.

## 8. SHAP explainability

- **Explainer:** `shap.TreeExplainer(model, model_output="raw")` for XGBoost
  (`LinearExplainer` with a 200-row training background for logistic regression).
- **Scale:** SHAP values are in **log-odds**. For every customer
  `base_value + Σ SHAP = logit(p)`; the pipeline asserts this (max error 2.6e-06).
  A contribution of +0.5 multiplies the odds by e^0.5 ≈ 1.65; it is **not** +0.5 or +50 percentage
  points of probability. The base value −1.02 corresponds to ≈26.4%.
- **Feature mapping:** SHAP is computed on the 45 encoded columns (4 numeric + 41 one-hot), then the columns of each
  original variable are summed (valid because SHAP values are additive), giving one contribution per
  original variable (e.g. `Contract`). `get_feature_mapping` checks that the mapping matches the
  preprocessor output order.
- **Scope:** global plots use the 1,409 test customers; waterfalls explain four test customers chosen
  by predicted probability (highest, 90th percentile, nearest to the threshold, lowest).

**Limitations of the explanations.** SHAP explains the *model*, not the world: it says which
inputs moved this prediction, not what would happen if the company changed them. Correlated features
(`tenure`, `TotalCharges`, `Contract`) share credit in ways that depend on the tree structure, and
the tree-path-dependent explainer follows the data distribution, so attributions between correlated
variables should be read together.

## 9. Reproducibility

- Exact library versions pinned in `requirements.txt`; seeds fixed (`random_state=42`).
- `python main.py` regenerates models, metrics, figures, SHAP outputs and predictions; two full
  runs produced identical metrics.
- `models/model_metadata.json` stores the selected model, parameters, threshold, risk bands, test IDs,
  training timestamp and library versions.
