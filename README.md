# Bank Account Fraud Detection

Predicting fraudulent bank account applications with scikit-learn's `HistGradientBoostingClassifier`, using a time-based split and a documented set of data-cleaning and modeling decisions.

**Final result (test months 6-7):** ROC-AUC **0.891**, PR-AUC **0.186**, about 13x the no-skill baseline for PR-AUC.

---

## 1. The question

Given the information available when someone applies for a bank account, can we rank applications by how likely they are to be fraudulent?

The goal is to **rank** applications (and so flag the riskiest ones for review), not to block them automatically. That choice drives the metrics used below.

## 2. Data

- **Source:** Bank Account Fraud dataset (NeurIPS 2022, `Base.csv`). *Add the Kaggle link and license here.*
- **Size:** 1,000,000 rows, 32 columns, 8 months (`month` 0-7).
- **Target:** `fraud_bool`. 11,029 fraud cases, a base rate of **1.10%**.
- **No true nulls.** Missing values are hidden as placeholder numbers (see section 4).
- **Not committed to this repo** because of file size. Download it from the source above and place it in the project folder.

### Fraud rate drifts upward over time

| Month | Rows | Fraud cases | Fraud rate |
|---|---|---|---|
| 0 | 132,440 | 1,500 | 1.13% |
| 1 | 127,620 | 1,198 | 0.94% |
| 2 | 136,979 | 1,198 | 0.87% |
| 3 | 150,936 | 1,392 | 0.92% |
| 4 | 127,691 | 1,452 | 1.14% |
| 5 | 119,323 | 1,411 | 1.18% |
| 6 | 108,168 | 1,450 | 1.34% |
| 7 | 96,843 | 1,428 | 1.47% |

This drift is the main reason for splitting by time (section 3).

## 3. Methodology

### Time-based split

| Piece | Months | Rows | Fraud cases | Fraud rate |
|---|---|---|---|---|
| Train | 0-5 | 794,989 | 8,151 | about 1.03% |
| Test | 6-7 | 205,011 | 2,878 | about 1.40% |

A deployed fraud model learns from the past and scores the future, so the test set must come after the training set. A random split would mix months and hide the drift above. `month` is used only to split, never as a feature.

For tuning decisions, the training months were split again: **train on months 0-4, validate on month 5** (1,411 fraud cases). The test months were kept for one final score.

### Metrics

- **ROC-AUC:** the chance the model ranks a random fraud above a random legitimate application.
- **PR-AUC:** average precision. This is the harsher metric for rare positives.
- **Accuracy was never used.** Predicting "never fraud" scores about 99% accuracy while catching nothing.
- **Baseline:** a `DummyClassifier(strategy='prior')` gives ROC-AUC 0.5, and a PR-AUC equal to the fraud rate of the evaluation set (about 0.014 on the test months, 0.0118 on month 5). Scores are read as a multiple of that floor, because PR-AUC depends on the base rate and can't be compared across different sets.

## 4. Data cleaning decisions

### Columns removed from the features

| Column | Reason |
|---|---|
| `fraud_bool` | The target |
| `month` | Used only for the time split |
| `device_fraud_count` | Constant 0 for every row, so it carries no information |
| `customer_age` | Sensitive attribute with no measurable benefit (see section 6) |

All remaining features are known at application time, so there is no obvious leakage column.

### Hidden missing values

| Column | Placeholder | Share of rows | Treatment |
|---|---|---|---|
| `prev_address_months_count` | -1 | 71.3% | Converted to NaN |
| `bank_months_count` | -1 | 25.4% | Converted to NaN |
| `current_address_months_count` | -1 | 0.43% | Converted to NaN |
| `session_length_in_minutes` | -1 | 0.20% | Converted to NaN |
| `device_distinct_emails_8w` | -1 | 0.04% | Converted to NaN |
| `intended_balcon_amount` | Negative values | 74.3% | Converted to NaN (see limitations) |
| `velocity_6h` | Negative values | 0.004% (44 rows) | Converted to NaN |
| `credit_risk_score` | Negative values | 1.4% | **Left unchanged** |

**Why NaN and not -1 or deletion:** leaving -1 would tell the model that 71% of applicants have an address history shorter than zero months. Dropping rows would discard most of the data. `HistGradientBoostingClassifier` handles NaN natively and can use missingness as information.

**Why `credit_risk_score` was left alone:** its 14,445 negative rows showed no dominant value (the most common, -1, appears only 488 times) and spread continuously from -170 to -1. That looks like the low end of a genuine score, not a placeholder. These rows also have a fraud rate of 0.39%, about a third of the overall rate, so they carry real signal.

### Categorical columns

Five text columns (`payment_type`, `employment_status`, `housing_status`, `source`, `device_os`) have 2 to 7 values each. They are anonymized codes with no natural order, so ordinal encoding was ruled out. They are converted to pandas `category` dtype, with the category list **learned from the training data only**, and the model reads them natively.

Fraud rate by category showed real signal in four of the five columns:

| Column | Strongest pattern |
|---|---|
| `housing_status` | `BA` has a 3.7% fraud rate (3.4x overall) and holds 58% of all fraud |
| `device_os` | `windows` has a 2.5% rate (2.2x overall) and holds 59% of all fraud |
| `employment_status` | `CC` at 2.5%, while `CF`, `CE`, and `CD` sit at 0.2-0.4% |
| `payment_type` | `AC` at 1.7%, `AA` at 0.5% |
| `source` | Weak: `TELEAPP` at 1.6% on only 7,048 rows |

**Rare categories were kept unmerged.** `AE` (289 rows, 1 fraud case), `CG` (453 rows, 7 cases) and `BG` (252 rows, 1 case) have too few fraud cases for their rates to mean anything. Each appears in both train and test, so none is new to the model at test time. Merging a lone rare category into "OTHER" would only rename it. Their rates should not be trusted, and the model's default minimum leaf size limits how far it can overfit to them.

## 5. Model

`HistGradientBoostingClassifier` with:

- `categorical_features='from_dtype'`
- `class_weight='balanced'`
- `random_state=42`
- all other settings at their defaults

No hyperparameter tuning was done.

### Choosing `class_weight`

Fraud is about 1% of rows. `'balanced'` makes the total weight of the two classes equal, so each fraud row counts roughly 90 times more than a legitimate one.

| Setting | Validation ROC-AUC (month 5) | Validation PR-AUC | PR-AUC vs. floor (0.0118) |
|---|---|---|---|
| `None` | 0.8885 | 0.1600 | about 13.5x |
| `'balanced'` | **0.8936** | **0.1854** | about 15.7x |

The gaps (about 0.005 ROC-AUC, 0.025 PR-AUC) are well above the run-to-run noise of about 0.001, which was measured by rerunning with different seeds. The same ordering appeared on the test months in an earlier comparison (ROC-AUC 0.8886 vs 0.8933, PR-AUC 0.1762 vs 0.1895), so `'balanced'` was chosen.

## 6. Experiment: is `customer_age` needed?

`customer_age` is one of the dataset's sensitive attributes. In validation runs with different seeds, removing it changed scores by about 0.001 or less. On the final test months the cost was slightly larger:

| | Test ROC-AUC | Test PR-AUC |
|---|---|---|
| With `customer_age` | 0.8933 | 0.1895 |
| Without `customer_age` | 0.8909 | 0.1856 |

Removing it cost about 0.002 ROC-AUC and 0.004 PR-AUC, a few times the measured noise. The model was trained without it anyway, because it is a sensitive attribute and the performance cost is small. Whether the cost is real signal or split-specific noise was not tested further.

## 7. Final results

Model trained on months 0-5 and scored **once** on months 6-7, with no changes made after seeing the test score.

| | ROC-AUC | PR-AUC |
|---|---|---|
| No-skill baseline | 0.500 | about 0.014 |
| **Final model** | **0.891** | **0.186** |

The model's PR-AUC is about 13 times the baseline. Given a 1.4% fraud rate and no tuning, this shows the model separates fraud from legitimate applications well but would not be precise enough to act on alone.

## 8. Limitations and future work

- **Probabilities are inflated.** `class_weight='balanced'` makes the model behave as if fraud were common, so a score of 0.7 is not a 70% chance of fraud. Scores are good for ranking, but not for reading as probabilities or applying a fixed cutoff.
- **No threshold analysis yet.** Precision and recall at a chosen cutoff, and recall at a fixed false-positive rate (for example 5%), were not computed.
- **`intended_balcon_amount` assumption.** Its negative values were treated as missing, which is a hypothesis based on the dataset's documentation, as I recall it. The model was not compared against leaving the raw values in.
- **Fairness is untested.** Removing `customer_age` does not remove its influence, because columns such as `income` and `employment_status` can act as proxies. Flag rates by age group were not compared.
- **No feature importance.** `permutation_importance` was not run, so which features drive the model is not established beyond the category analysis above.
- **Drift.** Fraud rose from about 0.9% to about 1.5% across the eight months, so performance would likely degrade without retraining.
- **Single seed on the test set.** The final score comes from one run.

## 9. What this project practiced

Time-based validation, spotting hidden missing values, checking whether rare categories hold enough evidence to trust, choosing a setting on a validation slice instead of the test set, reading PR-AUC against its baseline, and reporting a small trade-off honestly.

## 10. How to reproduce

1. Download `Base.csv` from the source above and place it in the project folder.
2. Install: `pandas`, `numpy`, `scikit-learn`.
3. Run the notebook top to bottom. It loads the data, applies the cleaning rules in section 4, splits by `month`, encodes the categories using the training months only, trains the model, and scores the test months once.
