# CTR Prediction MLOps Pipeline (Criteo 1TB Click Logs — Day_2)

End-to-end **MLOps-style CTR (Click-Through Rate) prediction pipeline** built on a large-scale slice of the **Criteo 1TB Click Logs** dataset.  
This repo focuses on building a production-realistic workflow using **one day partition (Day_2)** due to compute constraints, while keeping the architecture scalable to multiple days.

---

## Project Goals

- Build a **reproducible ML pipeline** on a large dataset (~**20M rows**, ~**66GB** stored in DB across extracted days)
- Perform **EDA at scale** (in-database aggregation, bucketing analysis)
- Implement **production-aligned preprocessing**
  - Numeric imputation + stabilization for heavy-tailed distributions
  - Categorical missing/rare handling
- Train a **baseline CTR model** (LightGBM) and extract **feature importance**
- Produce artifacts that can later extend to:
  - feature store
  - model registry
  - monitoring/drift detection
  - automated retraining

---

## Dataset

**Criteo 1TB Click Logs**  
- Each row = an **ad impression**
- `label` = 1 if clicked, 0 otherwise
- `int_feature_1..13` = anonymized numeric signals (heavy-tailed, sparse)
- `cat_feature_1..26` = anonymized hashed categorical IDs (high-cardinality)

> Note: Features are anonymized and do not map to human-readable “user demographics”. They are engineered signals/IDs commonly found in real ad systems.

---

## Storage / Environment

- **Database:** Postgres (partitioned by `day`)
- **RAM:** 16GB (hence, day-by-day processing and in-database EDA)
- **Working partition:** `day = 'day_2'` (≈ 20M rows)

---

## Preprocessing Decisions (Why these choices?)

### Numeric Features (`int_feature_*`)
CTR datasets contain heavy-tailed numeric features (a lot of small values, a few extremely large values).  
To stabilize learning:

1. **Impute missing numeric values with `0`**
   - In ad logs, missing often means *no signal available*.
2. **Log transform numeric values**
   - `x_clean = log(1 + x)`
   - Compresses large outliers, keeps small differences meaningful.
   - Ensures model training is more stable.

✅ Result: numeric values become easier for both tree models and deep models to learn from.

---

### Categorical Features (`cat_feature_*`)
Categoricals are hashed IDs with extreme cardinality. “Imputation” like mean/median does not apply.

- Missing categories are mapped to:
  - `MISSING` (true missing)
- Rare categories are mapped to:
  - `RARE` (low-frequency bucket)

This:
- prevents exploding vocabulary size,
- reduces noise from ultra-rare IDs,
- keeps “missingness” as an explicit signal.

---

## EDA Approach (Day-by-day, scalable)

Because the dataset is large, EDA is done primarily in Postgres using aggregation queries.

### 1) Basic CTR stats (per day)
- impressions = `COUNT(*)`
- clicks = `SUM(label)`
- CTR = `SUM(label) / COUNT(*)`

### 2) Numeric signal check via bucketing
For each numeric feature, values were bucketed and CTR was computed per bucket:

- `impressions per bucket`
- `CTR per bucket`

This helps validate whether CTR changes across value ranges (univariate predictive signal).

**Key observation:** some numeric features show strong monotonic or threshold patterns (good predictive signal).

---

## Baseline Model

### Model
- **LightGBM** (baseline CTR model)
- Evaluated using:
  - AUC
  - LogLoss

### Feature importance
We use **Gain-based feature importance** (how much each feature reduces loss).

Top features (sample from Day_2 baseline):

| Rank | Feature          | Gain % (approx) |
|------|------------------|-----------------|
| 1    | int_feature_4    | ~9.88%          |
| 2    | int_feature_12   | ~7.40%          |
| 3    | int_feature_1    | ~4.94%          |
| 4    | int_feature_2    | ~4.84%          |
| 5    | int_feature_11   | ~4.66%          |
| 6    | cat_feature_26   | ~4.06%          |
| ...  | ...              | ...             |
| 39   | cat_feature_6    | ~0.12%          |

✅ This confirms that both **numeric** and **categorical** features carry strong signal in CTR modeling.

---

## Feature Selection Strategy (Justification)

We do **not** drop columns early.  
Instead, we follow a production-safe approach:

1. Train baseline with all features
2. Rank by **Gain%**
3. Propose dropping only the **lowest-importance tail**
4. Retrain and compare AUC/LogLoss

Drop candidates typically include features with very low gain (example: <0.5%), but removal is only accepted if validation metrics stay stable.

---
