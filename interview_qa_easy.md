# Interview Knowledge Guide — Part 3: Interview Questions (Easy)

For every question: the answer, why it's correct, what NOT to say, and
where in the project it's grounded.

---

**Q: Why did you choose this dataset?**

A: NSL-KDD is the standard benchmark for classical ML-based intrusion
detection — it's widely cited, so results are comparable to existing work,
and it fixes a real flaw in the original KDD'99 dataset (duplicate records
that let models trivially memorize repeated rows). It's also
pre-aggregated into connection-level features rather than raw packets,
which keeps the ML side classical and fully explainable, matching the
scope I wanted for this project.

Why correct: this is the actual, stated reason (README, Dataset section) —
not "it was the first one I found," which would undercut the deliberate
scoping decisions made throughout.

What not to say: "It's the most accurate dataset" (accuracy isn't a
property of a dataset) or implying it's modern/production traffic — it
isn't, and the Limitations section says so explicitly.

Grounded in: `README.md` → Dataset section.

---

**Q: Why is this a classification problem?**

A: The target is a category (normal or attack) with no numeric ordering
between the classes — regression predicts a continuous number, not a
label, so classification is the only fit here.

Why correct: this is the basic supervised-learning distinction; getting it
wrong signals a fundamental gap.

What not to say: Don't confuse this with "binary classification" being the
only option — the project could have modeled `attack_category` as a
5-class problem too (and that's a documented Future Improvement); the
current model does binary only.

Grounded in: `src/load_data.py` (creation of `label_binary`), README →
Dataset.

---

**Q: What is data leakage? How did you prevent it?**

A: Data leakage is when information from outside the training set —
especially the test set — influences training or preprocessing, making
evaluation look better than it would in reality. I prevented it in two
places: (1) the encoder/scaler is `fit()` only on the training set, then
used to `.transform()` the test set, not fit on combined data; (2)
hyperparameter tuning uses cross-validation within the training set only —
the test set is touched exactly once, for final reported metrics.

Why correct: this is the textbook definition, and I can point to the exact
lines that implement the fix.

What not to say: Don't just say "I used train/test split" — that alone
doesn't prevent leakage if the *preprocessing* was fit on combined data
before the split was used. The fix has to be at the fit/transform level.

Grounded in: `src/preprocess.py` (`fit_transform` on train, `transform`
only on test), `src/tune.py` (cross-validation, docstring explains why).

---

**Q: Why did you use a train/test split?**

A: To get an honest estimate of how well the model generalizes to data it
hasn't seen. Evaluating on the same data used for training measures
memorization, not real performance.

Why correct: this is the fundamental purpose of held-out evaluation.

What not to say: Don't say "to make the code run faster" or anything
implying the split's purpose is computational rather than statistical.

Grounded in: NSL-KDD ships as separate `KDDTrain+.txt` / `KDDTest+.txt`
files, used directly as train/test in `src/load_data.py`.

---

**Q: Why not use accuracy as your main metric?**

A: The test set is imbalanced (57% attack, 43% normal) and the two error
types have very different real-world costs — missing an actual attack
(false negative) is typically worse than a false alarm (false positive).
Accuracy treats both errors equally and can look deceptively good even
when a model is bad at catching the minority behavior that actually
matters. I used precision, recall, F1, and ROC-AUC instead, and reported
per-attack-category recall specifically because it reveals weaknesses the
aggregate numbers hide.

Why correct: this is exactly the reasoning documented in the README's
Evaluation section, backed by real numbers (76% accuracy vs. only 60%
recall on attacks).

What not to say: Don't claim accuracy is "useless" — it's still reported
and has some value; the point is it shouldn't be the *only* or primary
metric here.

Grounded in: `README.md` → Evaluation.

---

**Q: What does precision mean in this project?**

A: Of everything the model labeled "attack," what fraction actually was an
attack. My tuned model has 0.97 precision on the attack class — when it
raises an alarm, it's right 97% of the time.

Why correct: this is the exact metric definition applied to this project's
actual measured number.

What not to say: Don't swap precision and recall — a very common mistake.
Precision is about the alarms you raised; recall is about the attacks that
existed.

Grounded in: `src/tune.py` output / README → Evaluation table.

---

**Q: What does recall mean in this project?**

A: Of all the attacks that actually happened in the test set, what
fraction the model caught. My tuned model has 0.60 recall overall on the
attack class — it misses 40% of actual attacks — and the per-category
breakdown shows that miss rate is much worse for R2L (94% missed) and U2R
(91% missed) specifically.

Why correct: matches the measured confusion matrix (5,110 missed / 12,833
actual attacks in test).

What not to say: Don't quote only the aggregate 60% without mentioning the
per-category gap — that's the more important, more honest finding.

Grounded in: `README.md` → Evaluation, per-category detection table.

---

**Q: What is overfitting?**

A: When a model learns patterns specific to the training data (including
its noise) rather than patterns that generalize, so it performs well on
training data but poorly on new data. Random Forests can overfit if trees
are allowed to grow very deep with no minimum sample constraints — which is
part of why `min_samples_split` and `min_samples_leaf` were tuned rather
than left at defaults.

Why correct: standard definition, tied directly to specific hyperparameters
in this project.

What not to say: Don't say "overfitting means the model is too accurate" —
overfitting is about the *gap* between training and test performance, not
about a model being "too good."

Grounded in: `src/tune.py` docstring (parameter explanations).
