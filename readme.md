
**Scalable CTR Prediction with DLRM: An End-to-End MLOps Pipeline from EDA to Deep Learning**:

## An End-to-End MLOps Pipeline from EDA to Deep Learning

---

# 📌 Project Overview

This project implements a **production-style CTR prediction system**, starting from raw compressed logs and evolving into a scalable Deep Learning Recommendation Model (DLRM) pipeline.

Unlike toy examples, this project includes:

* Raw data ingestion
* SQL-style exploratory analysis
* Statistical feature validation
* Imputation & transformation rebuild
* LightGBM baseline modeling
* Hyperparameter tuning
* CPU-optimized DLRM training
* Artifact versioning
* Modular MLOps-ready structure

Dataset size: ~20M+ rows
Hardware constraint: 16GB RAM, CPU-only

---

## Dataset
Data source :https://huggingface.co/criteo
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
- **Working partition:** `day = 'day_2','day_5` (≈ 20M rows)

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


# 🗂 Repository Structure

```
.
├── artifacts/
│   ├── eda_lgbm_metrics.json
│   ├── eda_lgbm_tuning_results.csv
│   ├── preprocess_artifacts.json
│
├── code_folder/
│   ├── eda_analysis.py
│   ├── analysing_data.ipynb
│   ├── loading_data.ipynb
│   ├── transform_pipeline.py
│   ├── transform_pipeline.ipynb
│   ├── imputation_rebuild.py
│   ├── train_dlrm_cpu.py
│
├── dataset/
├── dlrm_shards/
├── sql_analysis/
├── dockerfiles/
├── day_2.gz ... day_10.gz
├── readme.md
└── .gitignore
```

---

# 🏗 Pipeline Evolution

This project was built incrementally — mirroring real-world ML system development.

---

## 1️⃣ Raw Data Ingestion

Source:

* Compressed daily logs (`day_X.gz`)

Initial step:

* Load raw CTR logs
* Validate schema
* Inspect label distribution
* Verify data integrity

Implemented in:

* `loading_data.ipynb`

---

## 2️⃣ Exploratory Data Analysis (EDA)

Performed deep statistical analysis:

* CTR per feature bucket
* Numeric feature distribution analysis
* Heavy-tail detection
* Missing value behavior analysis
* Category frequency analysis

Validated signal strength via:

* Bucketed CTR comparison
* Statistical variation checks

Implemented in:

* `analysing_data.ipynb`
* `eda_analysis.py`

Artifacts:

* `eda_lgbm_metrics.json`
* `eda_lgbm_tuning_results.csv`

---

## 3️⃣ Feature Engineering & Imputation Rebuild

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

### 🔹 Numeric Features

* Missing → 0
* Log transformation applied
* Heavy-tail stabilization

### 🔹 Categorical Features

* Missing → "UNK"
* Rare categories handled
* Hash-based bucket encoding for memory safety

Implemented in:

* `imputation_rebuild.py`
* `transform_pipeline.py`

Artifacts:

* `preprocess_artifacts.json`

---

## 4️⃣ Baseline Model — LightGBM

Purpose:

* Establish performance baseline
* Validate feature importance
* Perform gain-based feature ranking

Used:

* Gain importance
* Split importance
* Hyperparameter tuning

Outcome:

* Identified strongest predictive features
* Reduced redundant features

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
## 5️⃣ Feature Selection Strategy

Instead of dropping columns blindly:

* Ranked by gain %
* Performed controlled pruning
* Validated AUC impact

Ensured:

* Signal retention
* Reduced model complexity
* Production defensibility

---

## 6️⃣ Sharded Dataset Strategy (Memory-Safe)

Given 16GB RAM:

* Split dataset into shards
* Stream one shard at a time
* Avoid full dataset in memory

Stored in:

* `dlrm_shards/`

---

## 7️⃣ Deep Learning Model — CPU-Based DLRM

Implemented in:

* `train_dlrm_cpu.py`

Architecture:

### 🔹 Embedding Layer

* One embedding table per categorical feature
* Hash bucket encoding

### 🔹 Bottom MLP

Dense features → embedding space projection

### 🔹 Feature Interaction

Pairwise dot-product interactions

### 🔹 Top MLP

Final probability prediction

Loss:
Binary Cross Entropy

Optimizer:
Adam

---

# 📊 Evaluation Metrics

CTR datasets are highly imbalanced.

Primary metric:

### ✅ ROC-AUC

Measures ranking quality.

Secondary:

* LogLoss
* Precision
* Recall
* F1

Accuracy intentionally not used.

---

# ⚙ Hyperparameter Search

Performed grid search over:

* Embedding dimension
* Learning rate
* Dropout
* Batch size

Selected best configuration using validation AUC.

---

# 🧠 MLOps Design Principles Applied

✔ Modular pipeline
✔ Artifact tracking
✔ Reproducible preprocessing
✔ Memory-aware training
✔ Feature schema freezing
✔ Experiment logging
✔ Baseline comparison before deep model

---

# 🔄 End-to-End Flow

```
                    ┌──────────────────────────────┐
                    │        Raw CTR Logs (.gz)    │
                    │  day_2.gz ... day_10.gz      │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │      Data Loading Layer      │
                    │  - Schema validation         │
                    │  - Label distribution check  │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │        EDA & Analysis        │
                    │  - Bucket CTR analysis       │
                    │  - Heavy-tail detection      │
                    │  - Missing value impact      │
                    │  - Feature signal validation │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │  Feature Engineering Layer   │
                    │  - Numeric stabilization     │
                    │  - Rare category handling    │
                    │  - Hash bucket encoding      │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │   LightGBM Baseline Model    │
                    │  - Feature importance        │
                    │  - Gain ranking              │
                    │  - Hyperparameter tuning     │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │ Feature Selection Strategy   │
                    │  - Controlled pruning        │
                    │  - AUC validation checks     │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │   Sharded Dataset Layer      │
                    │  - Memory-safe partitions    │
                    │  - Streaming IterableDataset │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │        DLRM Model            │
                    │  - Embeddings                │
                    │  - Bottom MLP                │
                    │  - Interaction Layer         │
                    │  - Top MLP                   │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │   Metrics & Artifacts        │
                    │  - AUC / LogLoss             │
                    │  - Model weights             │
                    │  - Hyperparameters           │
                    └──────────────────────────────┘

```