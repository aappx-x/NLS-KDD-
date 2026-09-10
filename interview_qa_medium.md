# Interview Knowledge Guide — Part 4: Interview Questions (Medium)

These test whether you actually understand the project, not just memorized
it.

---

**Q: Why Random Forest specifically? Why not XGBoost or another gradient
boosting model?**

A: Random Forest gets me nearly all the practical benefit — strong
performance on tabular data, robustness to unscaled/skewed features, and
free, honest feature importances — while being simpler to explain and
tune. XGBoost (gradient boosting) often edges out Random Forest slightly
on benchmarks by building trees sequentially, each correcting the previous
ones' errors, but that sequential dependency also makes it more prone to
overfitting if not carefully tuned, and it has more hyperparameters
(learning rate, subsample ratio, regularization terms) that I'd be tuning
somewhat blindly rather than genuinely understanding. Given this project's
explicit goal — depth and honesty over marginal performance gains — Random
Forest was the right call. I'd consider XGBoost a reasonable next step if
I wanted to push performance further and was willing to invest the study
time to understand it properly.

Why correct: this directly answers "why not X" by acknowledging X's real
advantage while giving a genuine, non-defensive reason for the choice made.

What not to say: Don't claim Random Forest is simply "better" than
XGBoost — it usually isn't, on average, in benchmarks. The honest reason
is explainability and scope, not raw performance.

Grounded in: `README.md` → Models section ("Chosen over a neural network
because...") — the same reasoning extends to gradient boosting.

---

**Q: Why not use a neural network?**

A: The dataset is tabular, moderate-sized (126k rows, 121 features after
encoding) — exactly the regime where classical tree ensembles reliably
match or beat neural networks, without needing GPU infrastructure,
careful architecture design, or the much larger datasets neural nets
typically need to shine. A neural net would also be much harder for me to
explain honestly in an interview — I'd be naming layers and activation
functions without being able to reason about what the network actually
learned, which conflicts directly with this project's interpretability
goal.

Why correct: this is a real, well-supported claim in the tabular-data ML
literature (tree ensembles are frequently competitive with or better than
deep learning on structured/tabular data of this size), not just a
convenient excuse.

What not to say: Don't say neural networks "can't do classification" or
"don't work on this kind of data" — they can, they're just not clearly
better here and would cost interpretability and honesty for a marginal
(if any) performance gain.

Grounded in: `README.md` → Models, project brief's explicit instruction to
avoid deep learning unless clearly justified.

---

**Q: Why did you use these specific features? Did you do feature
selection?**

A: I used all 40 non-constant raw features (dropping only
`num_outbound_cmds`, which was constant across every row) rather than
hand-picking a subset upfront, then let Random Forest's and permutation
importance analysis show which features actually mattered after training.
This is more defensible than guessing which features to drop beforehand —
I confirmed empirically (not just by assumption) that features like
`dst_bytes`, `src_bytes`, `flag_SF`, and the error-rate family dominate
the model's decisions, and that a feature I might have assumed was
important didn't necessarily show up as such.

Why correct: this reflects the actual pipeline — preprocessing dropped
only the genuinely useless column, and importance analysis (not manual
selection) is what identified which features matter.

What not to say: Don't claim you did rigorous upfront feature selection
(e.g. recursive feature elimination) if you didn't — the project's actual
feature selection is post-hoc interpretability, and that distinction
matters if pressed on it.

Grounded in: `src/preprocess.py` (`DROP_COLS`), `src/feature_importance.py`.

---

**Q: How did you handle class imbalance?**

A: The binary class balance is fairly mild (53.5%/46.5% in train), so I
used a light correction — `class_weight="balanced_subsample"` in the
Random Forest, which reweights each bootstrap sample so errors on the
minority class count more during training — rather than something heavier
like SMOTE oversampling, which would add complexity not justified by this
level of imbalance. I did not apply this to the underlying attack-category
imbalance (R2L: 995 rows, U2R: 52 rows, out of ~126,000) directly, and
that's reflected honestly in the poor recall on those categories — it's a
documented limitation and a listed future improvement, not something I
solved.

Why correct: matches exactly what `class_weight="balanced_subsample"` does
mechanically, and is honest about what wasn't addressed.

What not to say: Don't claim you "solved" class imbalance — the R2L/U2R
numbers in the Evaluation section directly show you didn't, and claiming
otherwise would be caught immediately by anyone who reads the results
table.

Grounded in: `src/train.py` / `src/tune.py` (`class_weight` parameter),
`README.md` → Future improvements.

---

**Q: How did you tune the model? Walk me through it.**

A: I used `RandomizedSearchCV` with 3-fold cross-validation on the
training set, searching over 5 hyperparameters (`n_estimators`,
`max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`),
sampling 10 random combinations rather than trying every combination
exhaustively (which would be hundreds of fits). I scored on F1 rather than
accuracy, since F1 balances precision and recall directly. The search
found `n_estimators=157, max_depth=None, min_samples_split=13,
min_samples_leaf=1, max_features='sqrt'` with a cross-validated F1 of
0.998 on training folds — then I evaluated that exact model, untouched,
on the held-out test set, where it got 0.963 ROC-AUC.

Why correct: this matches the actual code and actual numbers exactly — a
strong signal you ran this yourself rather than describing a generic
process.

What not to say: Don't quote the 0.998 CV F1 score as if it were the test
performance — that's a training-fold number and conflating it with test
performance would be a real, catchable error (and also worth noting
honestly: it's much higher than the ~0.75 test F1, which is itself worth
being able to explain — see the Hard question on this below).

Grounded in: `src/tune.py`, its printed output.

---

**Q: What happens if the model has high precision but low recall — what
does that mean practically for this system?**

A: It means when the system raises an alarm, it's very likely a real
attack (97% of the time) — so analysts can trust alerts and alert fatigue
is low — but the system is quietly missing a large fraction of actual
attacks (40% overall, and the vast majority of R2L/U2R attacks
specifically). In a real deployment, that's a serious risk: the system
would look reliable (few false alarms) while actually leaving a lot of
attacks undetected, which could create false confidence. Practically, this
is exactly the kind of finding that argues for lowering the decision
threshold below 0.5 to trade some precision for higher recall, especially
for rarer/higher-severity attack types.

Why correct: directly and correctly interprets the actual confusion
matrix and the ROC-AUC/accuracy gap discussed in the README.

What not to say: Don't say "high precision is always good enough" —
that's the exact wrong lesson from this project's results.

Grounded in: `README.md` → Evaluation (confusion matrix, ROC-AUC
discussion).

---

**Q: What does the confusion matrix tell you?**

A: `[[9448, 263], [5110, 7723]]` — reading it as [actual normal, actual
attack] rows against [predicted normal, predicted attack] columns: 9,448
normal connections were correctly identified as normal, 263 normal
connections were falsely flagged as attacks, 5,110 actual attacks were
missed (predicted normal), and 7,723 actual attacks were correctly caught.
Every precision/recall/F1 number is derived directly from these four
counts — for example, recall on attacks is 7723 / (7723 + 5110) = 0.60.

Why correct: shows you can read the raw matrix and derive the summary
metrics yourself rather than just quoting them.

What not to say: Don't mix up rows and columns — always state which axis
is actual vs. predicted before quoting numbers, since conventions differ
across tools.

Grounded in: `src/tune.py` output, `README.md` → Evaluation.

---

**Q: What are the limitations of your model?**

A: Several, stated directly rather than discovered under pressure: (1)
poor recall on R2L (6.3%) and U2R (9.0%) attacks due to extreme class
scarcity in training data; (2) trained on ~25-year-old simulated traffic
that doesn't reflect modern encrypted/cloud network patterns; (3) binary
output only — no attack-type classification in the deployed demo; (4) no
temporal modeling — each connection is scored independently with no
memory of prior connections from the same host, which real attacks
(especially slow scans) might exploit; (5) not tested against genuinely
novel attack types beyond what NSL-KDD's test set includes.

Why correct: this is a complete, specific list — not a vague "it could
always be better" answer — and matches the README's Limitations section
exactly.

What not to say: Don't give only one limitation and stop — a thin answer
here reads as either not having thought it through or trying to minimize
real weaknesses.

Grounded in: `README.md` → Limitations.
