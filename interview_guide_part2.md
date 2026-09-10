# Interview Knowledge Guide — Part 2: Cybersecurity & Deployment

(Continues from Part 1 — read that first.)

### Cybersecurity

**What is an IDS?**
An Intrusion Detection System monitors network or system activity and
raises alerts on suspicious behavior. It's *detective*, not *preventive* —
an IDS on its own doesn't block anything (that's the job of an Intrusion
Prevention System, IPS, or a firewall acting on the IDS's alerts). This
project is a prototype of the detection component only.

**Signature-based vs. anomaly-based IDS:**
Signature-based systems (like Snort in its classic mode) match traffic
against a curated database of known-attack patterns — fast and precise for
known attacks, but blind to anything not already in the database. Anomaly
(or ML-based) systems instead learn what "normal" looks like statistically
and flag deviations — in principle this can catch novel or slightly
modified attacks that don't exactly match a known signature, at the cost of
more false positives and being harder to explain to a SOC analyst ("why did
it flag this?"). Most real deployments use both together, and it's
important to say that in an interview rather than implying ML replaces
signatures.

**Why use ML at all, if signatures work?**
ML-based detection can generalize to variants of attacks it wasn't
explicitly told about, and can adapt as you retrain it on new data, without
someone manually writing a new rule for every variant. The trade-off is
interpretability and trust: a signature match is exact and explainable;
an ML flag is probabilistic and needs its own explanation tooling
(which is exactly why this project includes feature importance analysis).

**What is a false positive here?**
The model predicts "attack" but the traffic was actually normal (263 cases
in the test confusion matrix). Cost: analyst time spent investigating a
non-issue; too many of these and analysts start ignoring alerts entirely
("alert fatigue"), which is a well-documented real-world failure mode of
IDS deployments.

**What is a false negative here?**
The model predicts "normal" but the traffic was actually an attack (5,110
cases in the test confusion matrix, disproportionately R2L and U2R
attacks). Cost: a real intrusion goes undetected — in a production
context, this is usually the more expensive failure mode, which is the
core reason this project emphasizes recall over raw accuracy throughout.

**What are the limitations of ML-based IDS, generally — and specifically
in this project?**
General limitations: ML models can be evaded by attackers who deliberately
craft traffic to look statistically "normal" (adversarial evasion);
models can go stale as normal traffic patterns shift over time (concept
drift); models need labeled data to train on, which is expensive and slow
to produce for genuinely novel attacks. Specific to this project: the
severe class imbalance in R2L/U2R examples (995 and 52 rows) means the
model has very little to learn from for those categories, and it shows —
6.3% and 9.0% detection rates respectively. This project does not claim to
solve any of these limitations; it demonstrates and measures them.

### Deployment (as implemented in this demo — and what a real system would need)

**What happens when the Streamlit app receives an input?**
1. The user either loads one of 5 real preset examples (one per attack
   category, pulled directly from the NSL-KDD test set) or edits the
   feature values directly, including the full 40-feature table in the
   "advanced" expander.
2. On clicking "Run prediction," the app assembles a single-row DataFrame
   with the raw (unencoded, unscaled) feature values, in the same column
   order the preprocessor expects.
3. That row is passed through `preprocessor.transform()` — the *same*
   fitted `ColumnTransformer` object saved from training (loaded from
   `models/preprocessor.joblib`), not a freshly-fit one. This is essential:
   the input must be encoded/scaled using training-time statistics, or the
   numbers fed to the model would be meaningless relative to what it
   learned.
4. The transformed row is passed to `model.predict()` and
   `model.predict_proba()` — the same tuned Random Forest object saved
   from `src/tune.py` (loaded from `models/random_forest_tuned.joblib`).
5. The app displays the predicted class, the confidence (class
   probability), and a general permutation-importance chart (computed once
   ahead of time on the test set — not recomputed per-input, since that
   would be slow and isn't necessary to show which features the model
   relies on in general).

**How is the model loaded?**
Via `joblib.load()`, wrapped in Streamlit's `@st.cache_resource` decorator
so the model and preprocessor are loaded from disk once per app session,
not re-loaded on every button click or widget interaction (Streamlit
reruns the whole script on every interaction by default, so caching model
loading is a meaningful practical detail, not a nice-to-have).

**How does prediction happen (mechanically)?**
The 40 raw features become 121 encoded features via the fitted
preprocessor (one-hot columns for protocol/service/flag, scaled numeric
columns for everything else). Those 121 numbers are fed down all 157 trees
in the forest; each tree outputs a probability estimate; the forest
averages them into a final `predict_proba` output; a 0.5 threshold on that
probability determines the displayed normal/attack label.

**What would need to change for a real production IDS?** Be ready to list
these honestly rather than imply this demo already does them:
- Live traffic capture and flow aggregation (e.g. via NetFlow/IPFIX
  export, or tools like `scapy`/Zeek) instead of loading a static,
  pre-labeled dataset.
- Streaming inference at line rate, with attention to latency — this demo
  runs one row at a time, on demand, with no throughput requirement.
- Continuous retraining or drift monitoring, since network traffic
  patterns change over time and a model trained on 1990s simulated traffic
  will degrade in relevance.
- Alert routing/integration into a SIEM or SOC workflow, plus a feedback
  loop so analysts can mark false positives, which could be used to
  improve the model.
- Handling adversarial evasion — this demo assumes traffic is exactly like
  NSL-KDD, an assumption a real production system cannot make.
- Rethinking the R2L/U2R weakness specifically: more training data,
  different features (or additional data sources like process/system
  logs), or cost-sensitive learning that penalizes missing these rarer,
  higher-severity attacks more heavily during training.
