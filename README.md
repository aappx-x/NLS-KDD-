# Network Intrusion Detection — ML Prototype

A machine-learning classifier that flags network connection records as
**normal** or **attack**, trained and evaluated on the NSL-KDD intrusion
detection dataset. Built as a learning project combining classical ML with
core cybersecurity concepts — not a production security tool.

## Problem statement

Traditional intrusion detection systems (IDS) rely on signatures: known
patterns of malicious traffic. Signature-based systems can't catch attacks
that don't match an existing signature, and writing/maintaining rules for
every attack variant doesn't scale. Anomaly-based, ML-driven detection is a
complementary approach: a model learns statistical patterns that separate
normal traffic from attack traffic, potentially catching variants of known
attacks without an exact signature.

This project builds a simple version of that idea, end to end, and is
honest about what it can and cannot do.

## Motivation

This is a student project built for two goals: to genuinely learn classical
ML applied to a security problem,

## Dataset

**NSL-KDD** (an improved, de-duplicated version of the original 1998–99 KDD
Cup / DARPA intrusion detection dataset). Source files: `KDDTrain+.txt` /
`KDDTest+.txt`.


**Size:** 125,973 training rows, 22,544 test rows. 41 raw features per row
(one dropped as constant — see Preprocessing) plus the attack label.

**Target variable:** Each row's `label` names a specific attack (e.g.
`neptune`, `satan`, `guess_passwd`) or `normal`. This project uses:
- `label_binary` (0 = normal, 1 = attack) — what the trained model predicts.
- `attack_category` (normal / dos / probe / r2l / u2r) — a 5-way grouping of
  the ~39 specific attack names, used only for analysis, not as a model
  target in this version.

**Class distribution (train):** 53.5% normal, 46.5% attack.
**Class distribution (test):** 43.1% normal, 56.9% attack — the test set is
intentionally harder and more attack-heavy than train, and includes some
attack variants not present in training at all. This is a deliberate,
well-documented property of NSL-KDD, not a bug in this project's split.

**Attack category counts (train):**

| Category | Count | Description |
|---|---|---|
| normal | 67,343 | benign traffic |
| dos | 45,927 | denial-of-service (e.g. neptune, smurf) |
| probe | 11,656 | surveillance/scanning (e.g. satan, nmap) |
| r2l | 995 | remote-to-local (e.g. guess_passwd, ftp_write) |
| u2r | 52 | user-to-root privilege escalation (e.g. buffer_overflow) |


## Approach / ML pipeline

```
raw NSL-KDD .txt files
        │
        ▼
  src/load_data.py       — attach column names, derive label_binary / attack_category
        │
        ▼
  src/eda.py              — exploratory analysis, figures in notebooks/figures/
        │
        ▼
  src/preprocess.py        — drop constant column, one-hot encode categoricals,
        │                    scale numerics, fit only on train
        ▼
  src/train.py              — Logistic Regression (baseline) + Random Forest (primary)
        │
        ▼
  src/tune.py                — RandomizedSearchCV over Random Forest hyperparameters
        │
        ▼
  src/feature_importance.py   — Gini importance + permutation importance
        │
        ▼
  app/app.py                   — Streamlit demo using the fitted preprocessor + tuned model
```

## Preprocessing

- Dropped `num_outbound_cmds` — constant (always 0) across every row in the
  dataset, so it carries zero information.
- One-hot encoded `protocol_type` (3 values), `service` (70 values), `flag`
  (11 values) — 121 total encoded features from 40 raw ones.
- Standard-scaled numeric features. Random Forest doesn't strictly need
  this, but the Logistic Regression baseline does, and using one shared
  preprocessing pipeline for both models keeps the comparison fair and the
  pipeline simple.
- **Leakage prevention:** the scaler and encoder are `fit()` only on the
  training set, then used to `.transform()` the test set. Fitting on
  combined train+test data would leak test-set statistics into how training
  data gets represented — a subtle but real form of data leakage.
- `handle_unknown="ignore"` on the one-hot encoder: a few `service` values
  appear in the test set that never appear in train. Rather than crashing,
  these get encoded as all-zero — the realistic behavior for an unseen
  category at inference time.
- 80/20-style split is already provided by NSL-KDD (`KDDTrain+` /
  `KDDTest+`), so no additional split was performed. No validation set is
  carved out separately; hyperparameter tuning uses 3-fold cross-validation
  within the training set instead (see Hyperparameter Tuning).
- Class imbalance is mild (53/47 in train) and was not aggressively
  corrected. `class_weight="balanced_subsample"` was used in the Random
  Forest as a light adjustment rather than oversampling/SMOTE, which would
  have added complexity not justified by this level of imbalance.

## Models

**Baseline — Logistic Regression** (`sklearn.linear_model.LogisticRegression`,
`max_iter=1000`): a linear model; each of the 121 encoded features gets one
learned coefficient, and the sign/magnitude directly tells you whether that
feature pushes toward "attack" or "normal." Chosen as a baseline because it's
fast, fully transparent, and sets a floor that a more complex model has to
meaningfully beat to justify itself.

**Primary — Random Forest** (`sklearn.ensemble.RandomForestClassifier`): an
ensemble of decision trees, each trained on a bootstrap sample of the data
with a random subset of features considered at each split, with the final
prediction being a majority vote across trees. Chosen over a neural network
because:
- The data is tabular and moderate-sized — the regime where tree ensembles
  reliably match or outperform deep learning without needing GPU
  infrastructure, careful architecture search, or thousands of training
  epochs.
- It captures non-linear feature interactions Logistic Regression can't.
- It provides built-in, honest feature importances for free.
- It's robust to skewed numeric features (e.g. `src_bytes` has extreme
  outliers) without needing a log-transform first.

## Hyperparameter tuning

`RandomizedSearchCV` (not `GridSearchCV`) was used: with 5 tunable
parameters, an exhaustive grid search would require hundreds of model fits;
RandomizedSearchCV samples a fixed number (10) of random combinations from
the same search space and in practice finds a near-equally good
configuration for a fraction of the compute. 3-fold cross-validation,
scored on F1 (not accuracy — see Evaluation).

**Tuned parameters and why they matter:**

| Parameter | Role |
|---|---|
| `n_estimators` | number of trees; more = more stable predictions, diminishing returns |
| `max_depth` | how deep each tree can grow; unlimited depth risks overfitting to noisy rows |
| `min_samples_split` | minimum samples needed to split a node; higher = simpler trees |
| `min_samples_leaf` | minimum samples per leaf; higher = smoother, less overfit predictions |
| `max_features` | number of features considered per split; lower = more diverse trees |

**Best parameters found:** `n_estimators=157, max_depth=None,
min_samples_split=13, min_samples_leaf=1, max_features='sqrt'`
(best cross-validated F1: 0.998 — measured on training-set folds, not test).

Tuning used only the training set (via cross-validation); the test set was
never touched during the search. Touching the test set during tuning would
make the final reported test performance optimistic and not representative
of genuinely new data — the same leakage principle as the preprocessing
step, applied to model selection instead of feature scaling.

## Evaluation (measured on the held-out NSL-KDD test set)

| Model | Precision (attack) | Recall (attack) | F1 (attack) | Accuracy | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression (baseline) | 0.92 | 0.62 | 0.74 | 0.75 | 0.792 |
| Random Forest (untuned) | 0.97 | 0.62 | 0.75 | 0.77 | 0.962 |
| Random Forest (tuned) | 0.97 | 0.60 | 0.74 | 0.76 | **0.963** |

Confusion matrix, tuned Random Forest (rows = actual, columns = predicted;
order = [normal, attack]):

```
[[9448  263]
 [5110 7723]]
```

**Why accuracy alone is misleading here:** a model that always predicted
"attack" would score ~57% accuracy on this test set purely because attacks
are the majority class in the test split. Accuracy also treats a false
positive and a false negative as equally bad, which is not true for
security: **a false negative (missed attack) lets an intrusion through
undetected, while a false positive (false alarm) just costs analyst time
investigating benign traffic.** In most IDS contexts, missed attacks are
the more expensive error, which is why recall on the attack class — not
overall accuracy — is the number to watch first, and precision matters
mainly to keep alert fatigue manageable.

**Detection rate by attack category** (recall broken out per category,
which the aggregate metrics above hide):

| Category | Test rows | Detected as attack |
|---|---|---|
| dos | 7,460 | 77.2% |
| probe | 2,421 | 73.2% |
| r2l | 2,885 | **6.3%** |
| u2r | 67 | **9.0%** |
| normal | 9,711 | 97.3% correctly labeled normal |

This is the honest headline result of this project: the model is
genuinely strong on DoS and Probe attacks (which produce distinctive
traffic-volume and error-rate patterns) and on identifying normal traffic,
but it is **weak on R2L and U2R attacks**. Those attack types look
statistically close to normal traffic at the level of aggregated
connection-flow features (they're often a single login attempt or a local
privilege escalation, not a burst of unusual traffic), and NSL-KDD provides
very few training examples of them (995 and 52 rows respectively). This
matches known results in NSL-KDD literature and is not a coding error.

ROC-AUC of 0.963, notably higher than accuracy would suggest, reflects that
the model's predicted *probabilities* rank attacks vs. normal traffic well
even where the default 0.5 decision threshold misses cases — meaning
threshold tuning (biasing toward recall) is a real, available lever this
project doesn't currently exploit but easily could.

## Feature importance

Two methods were used and compared:

- **Gini importance** (built into Random Forest): top features were
  `dst_bytes`, `src_bytes`, `flag_SF`, `dst_host_srv_count`,
  `same_srv_rate`. This measures how much each feature reduced impurity
  across all tree splits, but is known to be biased toward high-cardinality
  features (like the one-hot encoded `service`, 70 columns).
- **Permutation importance** (test-set-based, cardinality-unbiased): top
  features were `src_bytes`, `dst_host_serror_rate`,
  `dst_host_same_src_port_rate`, `dst_host_srv_diff_host_rate`, `flag_SF`.
  This shuffles one feature at a time and measures the drop in F1 —
  a feature that matters gets a large drop.

**What these mean from a networking/security angle:** `src_bytes` /
`dst_bytes` (bytes sent/received) separate attacks that transfer unusually
little or unusually much data from typical sessions. `flag_SF` (connection
finished normally) versus other TCP flag states (e.g. `S0`, `REJ`) is a
strong normal-vs-attack signal because many DoS/probe attacks never
complete a full TCP handshake. The `serror_rate` / `rerror_rate` family
(SYN-error and REJ-error rates) captures scanning and flooding behavior —
a burst of connection errors to many ports or hosts is a classic
signature of both port scans and SYN-flood-style DoS attacks.

SHAP was deliberately not used — both importance methods above are
standard, fully explainable without extra machinery, and sufficient for
this project's interpretability requirement.

## Results summary

- Random Forest (tuned) is the final model: **97% precision, 60% recall on
  the attack class, ROC-AUC 0.963** on the NSL-KDD test set.
- Strong at detecting DoS and Probe attacks and normal traffic; weak on
  R2L and U2R — a known, explainable, and reported limitation of this
  approach and dataset combination.

## Screenshots

_Add screenshots of the running Streamlit app here:_
- `docs/screenshot_normal.png` — a normal-traffic prediction
- `docs/screenshot_attack.png` — an attack prediction with confidence and
  feature importance panel

## How to run

```bash
git clone <this-repo>
cd nids-project
pip install -r requirements.txt

# 1. Load and label the raw data
python src/load_data.py

# 2. (optional) Run EDA — figures saved to notebooks/figures/
python src/eda.py

# 3. Preprocess (fits and saves the encoder/scaler)
python src/preprocess.py

# 4. Train baseline + primary models
python src/train.py

# 5. (optional, slower) Hyperparameter tuning
python src/tune.py

# 6. Feature importance
python src/feature_importance.py

# 7. Launch the demo
streamlit run app/app.py
```

## Limitations

- Trained on an offline, ~25-year-old dataset of simulated traffic — not
  representative of modern encrypted/cloud traffic patterns.
- Binary classification only in the deployed demo (normal vs. attack); no
  attack-type classification is exposed in the app, though the data
  supports it as a future direction.
- Weak recall on R2L and U2R attack categories (see Evaluation) — this
  model would miss most attacks in those categories in practice.
- Not evaluated against previously unseen (zero-day) attack types beyond
  what NSL-KDD's test set already includes; no claim of zero-day detection
  is made.
- No temporal/sequential modeling — each connection record is scored
  independently, with no memory of prior connections from the same host.
- This is a prototype/demonstration, not a deployable or production-ready
  system: no live traffic capture, no alerting pipeline, no response
  automation.

## Future improvements

- Add attack-type (multi-class) prediction using `attack_category`.
- Address R2L/U2R class imbalance directly (e.g. targeted oversampling,
  cost-sensitive learning) rather than the light `balanced_subsample`
  weighting used here.
- Explore threshold tuning (moving off the default 0.5 cutoff) to trade
  some precision for meaningfully higher recall, given the security
  cost asymmetry discussed in Evaluation.
- Evaluate on a more modern intrusion detection dataset (e.g. CIC-IDS2017)
  to test how well the approach generalizes beyond NSL-KDD.
- Add basic packet/flow capture (e.g. via `scapy` or NetFlow exports) to
  move from "demo on saved data" toward "runs on locally captured traffic,"
  which would still fall short of production readiness but would be a
  meaningful next step.

## Tech stack

Python, pandas, NumPy, scikit-learn, matplotlib/seaborn, Streamlit.
