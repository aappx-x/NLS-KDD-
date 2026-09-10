# Interview Knowledge Guide — Network Intrusion Detection ML Prototype

This guide assumes you are studying this project from scratch. Read it in
order. Each layer builds on the one before it.

---

## Level 1 — 30-second explanation

Say something like this:

"I built a machine learning model that classifies network connections as
normal or malicious, using the NSL-KDD intrusion detection dataset. Each row
represents a summarized network connection with about 40 features — things
like bytes transferred, connection duration, and error rates. I trained a
Random Forest classifier, tuned its hyperparameters with RandomizedSearchCV,
and evaluated it with precision, recall, and ROC-AUC instead of just
accuracy, because in security a missed attack is usually worse than a false
alarm. The final model gets 97% precision and 0.96 ROC-AUC, but I can also
tell you where it's weak — it struggles with rarer attack types like
privilege escalation, and I understand why. I also built a small Streamlit
demo where you can enter connection features and get a live prediction."

**Why this answer works:** it's specific (real numbers), it's honest about
limitations before being asked, and it signals you understand evaluation
beyond accuracy — which immediately establishes credibility with a
technical interviewer.

---

## Level 2 — 2-minute explanation (full architecture)

**The problem being solved:** intrusion detection systems typically work by
signature matching — comparing traffic against a database of known-attack
patterns. Signature systems can't catch attacks that don't exactly match an
existing signature. Anomaly/ML-based detection is a complementary approach:
train a model on labeled traffic to learn statistical patterns that
separate normal from malicious traffic, which can, in principle, catch
variants of known attacks.

**The dataset:** NSL-KDD, a cleaned-up, widely-used benchmark version of the
original 1999 KDD Cup intrusion detection dataset. 125,973 training rows,
22,544 test rows, each a pre-aggregated connection record (not raw packets)
with 41 features (I dropped one constant column, so 40 used) — a mix of
basic connection info (protocol, service, duration, bytes transferred),
content features (failed logins, whether root shell was obtained), and
traffic-rate features computed over 2-second windows (like how many
connections went to the same host or service, and what fraction had
errors).

**The pipeline:**
1. **Load & label** — attach real column names to the raw comma-separated
   files (they ship with no header), derive a binary target (`normal` vs
   `attack`).
2. **EDA** — check class balance, correlations, and look for constant or
   suspicious features before doing anything else.
3. **Preprocess** — one-hot encode the 3 categorical features (protocol
   type, service, TCP flag), scale the numeric ones, and critically, fit
   the encoder and scaler on the training set only, to avoid leaking test
   set information into how training data is represented.
4. **Model** — a Logistic Regression baseline, then a Random Forest as the
   primary model. I chose Random Forest over a neural network deliberately:
   this is tabular, moderate-sized data where tree ensembles are the
   standard, defensible choice, and I can actually explain how every part
   of a Random Forest works, which I couldn't confidently say about a
   neural net's internals.
5. **Tune** — RandomizedSearchCV with 3-fold cross-validation on the
   training set, optimizing F1 (precision/recall balance), never touching
   the test set during search.
6. **Evaluate** — precision, recall, F1, ROC-AUC, confusion matrix, and a
   per-attack-category breakdown, because the aggregate numbers hide that
   the model is much better at some attack types than others.
7. **Interpret** — both Random Forest's built-in Gini importance and
   permutation importance, cross-checked against each other, to explain
   *why* the model predicts what it predicts.
8. **Demo** — a Streamlit app that loads the fitted preprocessing pipeline
   and tuned model, lets you enter or load example connection records, and
   shows the prediction, confidence, and general feature importance.

**The honest result:** ~97% precision and 0.96 ROC-AUC overall, but a
per-category breakdown reveals the model correctly flags roughly 77% of
DoS attacks and 73% of Probe attacks, but only about 6% of R2L attacks and
9% of U2R attacks — because those attack types are extremely rare in the
training data (995 and 52 rows respectively, out of ~126,000) and look
statistically similar to normal traffic at the connection-flow level. I
consider identifying and explaining that gap to be as important a result as
the headline numbers.

---

## Level 3 — Deep technical understanding

### Dataset

**What is network-flow data?**
Network-flow data summarizes a connection between two hosts into a single
record of aggregate statistics — total bytes sent, connection duration,
number of packets, error counts — rather than storing every individual
packet. This is how tools like NetFlow/IPFIX summarize traffic on real
networks, and it's why NSL-KDD's approach translates conceptually (though
not in exact feature format) to real-world traffic monitoring. It's a
deliberate simplification: you lose packet-level detail (like payload
content) but gain a compact, fixed-size representation per connection that
is easy to feed into a classical ML model.

**What are the features?**
Four groups, and knowing the groups (not just the names) is what makes this
defensible:
- *Basic connection features* (9 features): `duration`, `protocol_type`
  (tcp/udp/icmp), `service` (http, ftp, telnet, etc. — 70 possible values),
  `flag` (the state the connection ended in, e.g. `SF` = normal completion,
  `S0` = connection attempt seen, no reply — a strong attack signal),
  `src_bytes`, `dst_bytes`, `land` (1 if source and destination are the same
  host/port — a classic DoS trick), `wrong_fragment`, `urgent`.
- *Content features* (13 features): things that require looking "inside"
  the connection semantically — `hot` (number of "hot" indicators like
  accessing system directories), `num_failed_logins`, `logged_in`,
  `num_compromised`, `root_shell`, `su_attempted`, `num_root`,
  `num_file_creations`, `num_shells`, `num_access_files`,
  `num_outbound_cmds` (dropped — always 0), `is_host_login`,
  `is_guest_login`. These are the features most relevant to R2L/U2R attacks
  — but also the ones with the least data to learn from (see Evaluation).
- *Traffic features, "same host" 2-second window* (9 features): `count`
  (connections to the same host in the last 2s), `srv_count` (connections
  to the same service), and rate features derived from those
  (`serror_rate`, `rerror_rate`, `same_srv_rate`, `diff_srv_rate`,
  `srv_diff_host_rate`, etc.) — these are where DoS and Probe attacks leave
  their clearest signature, because scanning/flooding produces bursts of
  connections with unusual error or diversity patterns.
- *Traffic features, "same destination host" window* (10 features): the
  same idea but computed over the last 100 connections to the same
  destination host rather than a 2-second window — this catches slower,
  distributed patterns that the 2-second window would miss.

**What is the target?**
Two versions are available in this project: `label_binary` (0 = normal, 1 =
attack) is what the trained model actually predicts. `attack_category`
(normal/dos/probe/r2l/u2r) groups the ~39 specific attack names (like
`neptune`, `smurf`, `satan`, `guess_passwd`, `buffer_overflow`) into 5
top-level categories and is used only for analysis in this version of the
project — the model itself does not output an attack category.

**Why is this classification, not regression?**
The target is categorical (normal vs. attack, or one of 5 categories) with
no meaningful numeric ordering between classes — regression is for
predicting a continuous number (like bytes transferred), not a category
label. This is a textbook supervised classification problem.

### Preprocessing

**Why clean data?**
Garbage in, garbage out — a model can only be as reliable as the data it's
trained on. Concretely here: checking for missing values (there were none),
identifying and removing the constant `num_outbound_cmds` column (it can
only ever contribute noise, never signal, since it never varies), and
verifying the categorical columns didn't contain unexpected/malformed
values.

**Why encoding?**
`protocol_type`, `service`, and `flag` are text categories, and neither
Logistic Regression nor Random Forest can operate on raw strings — they
need numbers. One-hot encoding turns each category into its own binary
(0/1) column (e.g. `protocol_type_tcp`, `protocol_type_udp`,
`protocol_type_icmp`) rather than assigning arbitrary numbers like
tcp=0, udp=1, icmp=2, which would falsely imply an order or distance
between categories that doesn't exist.

**Why scaling?**
Logistic Regression's optimization (gradient descent under the hood) 
converges faster and more reliably when features are on comparable scales
— without scaling, a feature like `dst_host_count` (0–255) would dominate
the optimization compared to a rate feature like `serror_rate` (0–1) purely
because of its larger raw magnitude, not because it's more informative.
Random Forest doesn't need this (it splits on thresholds per feature
independently, so scale doesn't matter to it), but using one shared,
scaled feature matrix for both models keeps the pipeline simple and the
model comparison fair.

**Why split data?**
To get an honest estimate of how the model performs on data it has never
seen. If you evaluate a model on the same data it was trained on, you're
measuring memorization, not generalization. NSL-KDD already ships with a
separate train/test file, which is used here directly; hyperparameter
tuning further splits the training set into cross-validation folds so the
actual test set stays untouched until final evaluation.

**What is data leakage?**
Any way that information from outside the training data — especially from
the test set — influences the model during training or preprocessing,
making evaluation results look better than they'd be in reality. Two
concrete places this could have happened in this project, and how they
were avoided:
1. *Preprocessing leakage:* if the scaler/encoder were fit on train+test
   combined, statistics from the test set (its mean, its category
   presence) would subtly shape how training data gets represented. Fix:
   fit only on train, then `.transform()` (not `.fit_transform()`) the
   test set.
2. *Tuning leakage:* if hyperparameters were selected by checking
   performance on the test set (even just "peeking" once), the reported
   test score would be optimistic — you'd effectively be tuning to the
   test set. Fix: use cross-validation within the training set only for
   all tuning decisions; touch the test set exactly once, at the very end,
   for the final reported numbers.

### ML

**What is Logistic Regression?**
A linear model for classification. It computes a weighted sum of the input
features (each feature gets one learned coefficient/weight), then squashes
that sum through a sigmoid function to produce a probability between 0 and
1. If the probability is above 0.5 (by default), it predicts the positive
class (attack). It's called "regression" for historical reasons — it
predicts a continuous probability, then a threshold turns that into a
class label.

**What is Random Forest?**
An ensemble of decision trees. A single decision tree makes a series of
yes/no splits on feature values (e.g. "is `same_srv_rate` > 0.9?") to
partition the data into increasingly pure groups (mostly-normal or
mostly-attack). A single tree tends to overfit — it can memorize training
data by growing very deep. Random Forest fixes this by training many trees
(157, after tuning, in this project) on different bootstrap samples of the
data (sampling rows with replacement) and, at each split, only considering
a random subset of features rather than all of them (`max_features='sqrt'`
here). This "bagging + feature randomness" combination decorrelates the
trees from each other, so their individual errors tend to cancel out when
you average their votes — the forest as a whole is far more stable and
resistant to overfitting than any single tree.

**How does Random Forest actually work, step by step?**
1. Take the training set (N rows).
2. For each of the 157 trees: draw a bootstrap sample (N rows, sampled
   with replacement, so some rows repeat and some are left out).
3. Grow a decision tree on that sample. At each node, instead of
   considering all 121 encoded features to find the best split, only
   consider a random subset (with `max_features='sqrt'`, roughly
   sqrt(121) ≈ 11 features) and pick the best split among those.
4. Keep splitting until a stopping condition is hit (in the tuned model,
   `min_samples_split=13` means a node needs at least 13 samples to be
   split further; `min_samples_leaf=1` means a leaf can have as few as 1
   sample).
5. To predict a new row: run it down all 157 trees, each tree "votes"
   normal or attack, and the forest's prediction is based on the average
   predicted probability across all trees (which is also how
   `predict_proba` — the confidence score shown in the demo — is
   computed).

**Why is it suitable for this dataset?**
The dataset is tabular (rows and columns, not images/sequences/text),
moderate-sized (126k rows), and has genuine non-linear feature
interactions (e.g. a high `count` combined with a high `serror_rate` means
something different than either alone) that a linear model can't capture
without manual feature engineering. Tree ensembles are the standard,
well-benchmarked choice for exactly this kind of data, and they don't
require the scale of data or compute that a neural network would need to
reliably outperform them here.

**What are hyperparameters?**
Settings you choose before training that control how the model learns,
as opposed to parameters (like the learned coefficients in Logistic
Regression or the actual splits in each tree) which the model learns from
data. `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`,
and `max_features` are all hyperparameters — none of them are learned from
data directly; they're chosen (in this project, via RandomizedSearchCV) to
get the best cross-validated performance.

### Evaluation

**Accuracy** — fraction of all predictions that were correct. Simple, but
can be misleading when classes are imbalanced or when different error
types matter differently (both are true here).

**Precision** (for the "attack" class) — of everything the model flagged
as an attack, what fraction actually was an attack. High precision (0.97
here) means when this model raises an alarm, it's very likely to be
right — few false alarms.

**Recall** (for the "attack" class) — of everything that actually was an
attack, what fraction the model caught. 0.60 here means the model misses
40% of actual attacks overall (and far more within specific rare
categories — see the per-category breakdown in the README).

**F1-score** — the harmonic mean of precision and recall; a single number
that penalizes models which sacrifice one for the other. Used as the
tuning objective in this project specifically because it doesn't let a
model "cheat" by only optimizing one side of the precision/recall
trade-off.

**Confusion matrix** — a 2x2 table of actual vs. predicted class counts
([[9448, 263], [5110, 7723]] here — 9448 normal traffic correctly
identified, 263 normal traffic falsely flagged as attack, 5110 actual
attacks missed, 7723 actual attacks correctly caught). Every other metric
above is computed from these four numbers, so being able to read the
confusion matrix directly is a strong signal of real understanding.

**ROC-AUC** — measures how well the model's predicted probabilities rank
attacks above normal traffic across *all* possible decision thresholds,
not just the default 0.5 cutoff. 0.963 here is notably higher than the
accuracy (0.76) or recall (0.60) at the default threshold would suggest —
which tells you the model's underlying probability estimates are actually
quite good, but the default 0.5 cutoff isn't the best operating point for
this problem. This is itself an actionable insight: lowering the decision
threshold below 0.5 would trade some precision for meaningfully higher
recall, which — given that missed attacks are usually costlier than false
alarms — might be the right trade for a real deployment.

**Why might recall be more important than accuracy for an intrusion
detection system?** Because the cost of the two error types is
asymmetric. A false positive (flagging normal traffic as an attack) costs
an analyst some time investigating a false alarm. A false negative (an
actual attack that goes undetected) can mean a real breach goes unnoticed
— a much more expensive outcome in most contexts. Accuracy treats both
errors as equally bad and, on an imbalanced test set (57% attack here), a
model that mostly predicts the majority class can post a deceptively
decent accuracy while still missing a lot of attacks. Recall (and the
per-category recall breakdown) makes that risk visible in a way overall
accuracy hides.
