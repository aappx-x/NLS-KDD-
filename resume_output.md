# Resume Output

## 1. Project title
**ML-Based Network Intrusion Detection (NSL-KDD)**

## 2. Resume bullets (honest, based on actual measured results)

- Built a machine learning pipeline to classify network connections as
  benign or malicious using the NSL-KDD intrusion detection dataset
  (~126K training records, 41 flow-level features), including data
  cleaning, leakage-safe preprocessing, and a train/test evaluation
  achieving 0.97 precision and 0.96 ROC-AUC on held-out test data.
- Trained and compared Logistic Regression and Random Forest classifiers,
  then tuned the Random Forest via RandomizedSearchCV with cross-validation,
  and diagnosed a significant class-imbalance-driven weakness (6–9%
  detection rate on rare attack categories) through per-category
  evaluation rather than relying on aggregate accuracy alone.
- Applied Gini and permutation feature importance to interpret model
  decisions, identifying byte-transfer volume, TCP connection state, and
  connection-error rates as the dominant predictive signals, and connected
  those findings to their networking/security meaning.
- Developed an interactive Streamlit demo that loads the trained
  preprocessing pipeline and model to classify connection records in real
  time, with prediction confidence and feature-importance visualization,
  clearly scoped and labeled as a machine-learning demonstration rather
  than a production IDS.

*(Use 3 or 4 of these depending on resume space — the first two are the
strongest if you can only fit two.)*

## 3. Technology stack
Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, Streamlit, joblib.

## 4. One-line project description
A classical machine learning pipeline that classifies network traffic as
benign or malicious using the NSL-KDD dataset, with a Streamlit demo and
documented, honestly-evaluated performance and limitations.

## 5. GitHub README description (short, for the repo's about/description field)
ML-based network intrusion detection prototype (Random Forest + Logistic
Regression baseline) trained on NSL-KDD, with leakage-safe preprocessing,
tuned hyperparameters, interpretability analysis, and a Streamlit demo.
Evaluated with precision/recall/ROC-AUC and per-attack-category breakdown,
not just accuracy.

## 6. Skills / concepts you can legitimately claim after completing this project
- Supervised classification with scikit-learn (Logistic Regression, Random
  Forest)
- Data preprocessing: one-hot encoding, feature scaling, leakage-safe
  train/test pipeline design
- Hyperparameter tuning via RandomizedSearchCV with cross-validation
- Model evaluation beyond accuracy: precision, recall, F1, ROC-AUC,
  confusion matrix interpretation, per-class/per-category performance
  analysis
- Model interpretability: Gini (impurity-based) feature importance and
  permutation importance, and translating both into domain-specific
  (networking/security) explanations
- Core intrusion detection concepts: signature-based vs. anomaly-based
  detection, false positive/negative cost trade-offs in a security context
- Building and reasoning about a full ML pipeline end-to-end, from raw
  data to a working interactive demo
- Communicating model limitations and scope honestly — knowing what a
  prototype does and does not demonstrate, and being able to say so
  precisely rather than overclaiming

**Do not claim** (not demonstrated by this project): deep learning /
neural network expertise, MLOps or CI/CD pipelines, cloud deployment
experience, real-time streaming systems, production security engineering,
or zero-day/novel-attack detection capability.
