# Study Roadmap — This Exact Project

Follow this order. Each stage assumes the previous one is done. Don't skip
ahead to code before the concept stages — the goal is to be able to defend
*why*, not just recite *what*.

## Stage 0 — Prerequisites (if shaky, spend a day here first)
- Basic Python: functions, dictionaries, pandas DataFrames (selecting
  columns, filtering rows, `.groupby()`).
- What supervised learning is: features (X), labels/target (y), training
  vs. inference.
- What a percentage/rate means statistically (you'll need this fluently
  for precision/recall/error-rate features).

## Stage 1 — Understand the dataset before touching any model
- Read `README.md` → Dataset section fully.
- Open `data/processed/train.csv` yourself and scroll through 20–30 rows.
  Pick 5 rows at random and manually guess normal vs. attack before
  checking the label — this builds real intuition for what the features
  mean.
- Read `docs/interview_guide_part1.md` → "Dataset" subsection under Level 3.
- Run `python src/eda.py` yourself and look at every figure in
  `notebooks/figures/`. For each one, answer: what does this show, and why
  does it matter for building the model?

## Stage 2 — ML fundamentals needed for this project (not more than this)
- What classification is, and how it differs from regression.
- Logistic Regression: what a coefficient means, what a sigmoid does, why
  it's linear.
- Decision trees: how a single split works, what "impurity" means
  intuitively (a node is "pure" if all its rows are the same class).
- Random Forest: bagging (bootstrap sampling) + random feature subsets per
  split, and why that combination reduces overfitting versus one deep tree.
- Precision, recall, F1, confusion matrix, ROC-AUC — in that order, each
  building on the last. Don't move to ROC-AUC until precision/recall/the
  confusion matrix feel automatic.

## Stage 3 — Implementation, in pipeline order
Read source, run it, then re-read with the output in front of you.
1. `src/columns.py` + `src/load_data.py` — understand every derived column.
2. `src/eda.py` — re-run, re-read the printed correlation/constant-column
   output alongside the code that produced it.
3. `src/preprocess.py` — this is the file most likely to get deep
   follow-up questions (leakage). Read the docstring twice.
4. `src/train.py` — read the comments explaining *why* Random Forest, not
   just the `.fit()` call.
5. `src/tune.py` — understand every hyperparameter in the search space
   before running it; don't just treat it as a black box that improves
   numbers.
6. `src/feature_importance.py` — run it, then manually connect the top
   features back to the networking explanation in the README.
7. `app/app.py` — trace one prediction by hand: pick a preset, follow it
   through `preprocessor.transform()` conceptually, to `model.predict()`.

## Stage 4 — Cybersecurity concepts
- Read `docs/interview_guide_part2.md` fully.
- Be able to explain, unprompted, the difference between an IDS and an
  IPS, and between signature-based and anomaly-based detection.
- Be able to state, from memory, the real-world cost asymmetry between
  false positives and false negatives in a security context.

## Stage 5 — Interview preparation
- Go through `docs/interview_qa_easy.md`, `_medium.md`, `_hard.md` in
  order, out loud, without looking at the answer first. Only check the
  answer after attempting your own.
- Do a full run-through of the Level 1 (30-second) and Level 2 (2-minute)
  explanations from `docs/interview_guide_part1.md` until they sound like
  your own words, not a script.
- Mock-interview yourself (or a friend) using only the Hard-tier questions
  — if you can defend those fluently, the Easy/Medium tier will feel
  trivial by comparison.

## Final checkpoint
Before calling this "interview-ready," you should be able to, without
notes:
1. Explain the full pipeline end to end in under 2 minutes.
2. Read the confusion matrix out loud and derive precision/recall from it.
3. Explain why R2L/U2R recall is poor, using the actual numbers.
4. Say, unprompted, at least 3 things this project does NOT do.
