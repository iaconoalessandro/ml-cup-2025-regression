# Code & ML audit — ML-CUP 2025 regression pipeline

Audit of the regression pipeline (`regression/`) covering correctness of the MEE
implementation, data leakage across CV folds, and statistical validity of the
model-selection protocol. Findings are ordered by severity. Items marked
**FIXED** were corrected in this pass; items marked **FLAGGED** are
methodological observations left for the authors to decide on, because changing
them would change the reported results.

---

## A. Is the MEE implementation correct?

**Verdict: yes.** No action needed.

`compute_mee` is defined identically in `gridsearch_NN.py:239`,
`gridsearch_NN2.py`, and `ensembling.py:27`:

```python
np.mean(np.sqrt(np.sum((y_true - y_pred) ** 2, axis=1)))
```

This is the correct ML-CUP metric: the Euclidean norm of the residual per
sample across all 4 target dimensions, averaged over samples. It is fully
vectorised (no Python loop over samples), and there is no definition drift
between the three files.

`MEELoss` (the differentiable variant, `gridsearch_NN2.py:186`) is also
correct, and the `clamp(min=1e-12)` under the square root is the right guard —
`d/dx √x → ∞` as `x → 0`, which would otherwise produce an exploding gradient
whenever a prediction lands exactly on its target.

### Is MSE accidentally used for early stopping or model selection?

**No.** This was checked at every decision point:

| Decision point | Criterion used | Scale |
|---|---|---|
| Gradient descent (training loss) | MSE | standardised `y` |
| Early stopping | **MEE** | **original** |
| Per-fold score | **MEE** | **original** |
| CV aggregation / grid ranking | **MEE** | **original** |
| Stacking meta-feature quality | **MEE** | **original** |
| Final assessment | **MEE** | **original** |

Every reported number passes through `scaler_y.inverse_transform` before the
MEE is computed, so nothing is reported in standardised units. The split — MSE
optimises, MEE selects — is deliberate and documented in the file headers, and
it is applied consistently across **all** CV folds. No fold uses a different
criterion from another.

**One caveat worth knowing:** `LOSS_REGISTRY` offers `mse`, `huber`, and the
differentiable `mee`, but the shipped grid sets `COMMON_GRID['loss_fn'] =
['mse']` only. The MEE loss was implemented but never actually exercised in the
final search. That is a legitimate choice, but the write-up should not imply
MEE was the training objective — it was only ever the selection metric.

---

## B. Data leakage

### B1. Main pipeline — clean ✅

The path that produces the reported result is leak-free, and deliberately so:

- `cup_loader.py` splits **raw, unscaled** data. The 400/100 split happens
  before any transform touches the values.
- In `gridsearch_NN.py` / `gridsearch_NN2.py`, `StandardScaler` for both `X`
  and `y` is fit **inside** `train_one_fold`, on fold-training data only
  (`gridsearch_NN2.py:503-507`). The validation fold is only ever
  `transform`-ed.
- In the notebook's stacking stage (cell 65), the scalers **and** the PCA for
  KNN/SVR are fit inside each fold.
- The held-out 100 samples never enter any `fit`, any scaler, or any PCA.

### B2–B4. `ensembling.py` — three real leaks **FIXED**

This is an earlier prototype, superseded by the notebook, but it was shipping
with genuine leakage:

| # | Problem | Impact |
|---|---|---|
| **B2** | `StandardScaler` on `X` was fit on **all 400** training samples *before* the CV loop, then the folds were carved out of the already-scaled matrix. | Every OOF meta-feature was computed using statistics that saw the validation fold. Mild but real. |
| **B3** | The MLP's early stopping monitored the **outer validation fold** — the same rows whose predictions then became the OOF meta-features. | **The serious one.** The MLP's meta-features were optimistically biased while KNN's and SVR's were not, so the Ridge meta-learner systematically over-weighted the MLP. |
| **B4** | Fold MLPs used architecture `[272, 288, 144]`; the final MLP used `[256, 176, 80]`. | The meta-learner learned coefficients for one model and applied them to a different one. |

All three are fixed in `regression/ensembling.py` (see the `[L1]`–`[L3]`
markers). A fourth, non-statistical bug was fixed alongside: `criterion` was
used at the final training loop but defined only inside the fold loop — it
worked purely through Python's loop-variable scope leak and would have crashed
had the CV loop ever been skipped (`[L4]`).

The notebook had already independently fixed B3 with an inner 90/10 split; that
fix is what `run_pipeline.py` reproduces.

---

## C. Statistical fallacies in the tuning strategy

### C1. Early stopping on the same fold that reports the score — **FLAGGED**

In `gridsearch_NN.py` and `gridsearch_NN2.py`, `EarlyStopping` picks the best
epoch by minimising MEE on the validation fold, and then `best_val_mee` — the
minimum over ~100–500 epochs on that same fold — is what gets reported as the
fold's score.

Taking a minimum over hundreds of correlated evaluations on the same 80 samples
is an optimistic estimator. **The grid's absolute MEE figures are biased low.**

Two things make this less alarming than it sounds:

1. The bias applies uniformly to every configuration, so the *ranking* — which
   is all the grid is used for — stays usable.
2. The final reported number does **not** come from this path. The stacking
   stage re-estimates the MLP honestly, and the gap is visible and consistent:
   the grid claims **18.06** for the winning config, while the leak-free OOF
   estimate of the same model family is **18.95**. That ≈0.9 MEE difference is
   this bias, measured.

**Recommendation:** keep the grid numbers for ranking, but cite the OOF figures
when quoting MLP performance. Do not present 18.06 as an unbiased estimate.

### C2. ~1,500 configurations scored on one fixed set of folds — **FLAGGED, partly mitigated**

Every grid search reuses `KFold(shuffle=True, random_state=42)`. Across all
grids that is well over a thousand configurations scored against the *same five
partitions*, which is textbook multiple-comparison overfitting to the CV split:
the winner is partly selected for fitting these particular folds.

**The team already mitigated this correctly**, and it is worth calling out as a
strength. The multi-seed restart (notebook cell 63) re-ran the ten finalist
configurations across five seeds each and ranked on the *mean* rather than the
best. The winner changed as a result — from `256x256x128` (the single-grid
winner, C0320) to `256x256x256_adamw_heavy` (C0397). The original winner was
riding a lucky seed, and the protocol caught it. That is the right instinct,
executed properly.

### C3. The "VL" column in the report table is the meta-learner's in-sample error — **FLAGGED, quantified**

This is the subtlest issue and it propagates into the slides.

`RidgeCV` is fit on all 400 rows of OOF meta-features, and then
`stacking_oof_mee` is computed on **those same 400 rows**. The notebook comment
at cell 71 states the reasoning explicitly:

> "the OOF predictions are already out-of-fold, so the MEE computed here is an
> honest estimate even though we fit Ridge on the whole OOF set"

That does not follow. The base learners' *predictions* are out-of-fold, but the
**Ridge coefficients** are estimated on the same rows they are then scored on.
The stacking "VL" number is in-sample for the meta-learner.

**Quantified:** `run_pipeline.py` now computes both. On the fresh run, the
in-sample OOF figure is **14.30** while a properly nested CV of the meta-learner
gives **14.88** — a real optimistic bias of ≈0.6 MEE. The pipeline now reports
the nested figure in the VL column and keeps the in-sample one in
`metrics.json` under `stacking_insample` for comparability with the slides.

### C4. The epoch-rescaling heuristic double-counts — **FLAGGED**

For the final retrain, the median best epoch from CV (49, measured on 288-sample
inner-training sets) is multiplied by `400/288 ≈ 1.39` to give 68 epochs, with
the justification that "the model sees ~1.39× more data per epoch, so it needs
proportionally more steps to converge."

The premise argues the opposite way. `batch_size` is fixed at 32, so a
288-sample training set is 9 batches/epoch and a 400-sample one is 13. An epoch
on 400 samples **already contains 1.39× more gradient steps**. Multiplying the
epoch count by 1.39 as well means the final model takes ≈884 optimisation steps
against the ≈441 that CV validated — roughly **double**, not equal.

Step-matching would suggest ≈34 epochs, not 68. The cosine schedule is indexed
by epoch, which muddies a clean comparison, but the stated rationale does not
support the direction of the correction. It is a defensible engineering choice;
it is not the neutral one the comment presents it as.

### C5. `grid_results_optuna_gap_reduction.csv` has no source code — **FLAGGED**

`regression/results/` contains a CSV from an Optuna study, but `optuna` is
imported nowhere in the repository and no Optuna script is present. The result
is unreproducible from this repo. Either restore the script or drop the CSV.

---

## D. Factual errors in the presentation decks

Checked slide claims against the code and data. Details and fixes in
[`presentation/DECK_REVIEW.md`](presentation/DECK_REVIEW.md). Two are
significant:

1. **The target is 4-dimensional, not 2-dimensional.** `Github.pptx` states
   "a two-dimensional target" on slides 3 and 8. The data has 4 target columns
   (`df.iloc[:, 13:17]`), the network's output layer has 4 units, and the
   submission CSV has 4 prediction columns.

2. **"The meta-learner beats every member on both validation and test" is false
   on test.** Slide 19's own table shows KNN at **14.1284** and the stack at
   **14.3289** — the stack is 0.20 MEE *worse* than its best single member on
   held-out data. It does win on OOF. The reproduction run confirms the same
   pattern (KNN 14.13 vs stack 14.70), so this is a stable property of the
   model, not run-to-run noise.

This second point does not invalidate the work — a stack that ties its best
member while being far more stable across partitions is a perfectly reportable
outcome — but the claim as written is not supported by the table directly above
it.

---

## E. Reproducibility note

`run_pipeline.py` reproduces the deterministic components **exactly**:

| Model | Notebook | Fresh run | Match |
|---|---|---|---|
| KNN OOF | 15.1070 | 15.1070 | ✅ exact |
| KNN test | 14.1284 | 14.1284 | ✅ exact |
| SVR OOF | 15.5959 | 15.5963 | ✅ ~1e-4 |
| SVR test | 15.7672 | 15.7675 | ✅ ~1e-4 |
| MLP OOF | 18.9522 | 19.0131 | ≈ |
| Stacking test | **14.3289** | **14.7006** | ≈ |

The sklearn components reproduce bit-for-bit. The MLP does not, and the
stacking number inherits that drift. The cause is PyTorch version drift — the
notebook was run on an earlier release, this reproduction on torch 2.11 — where
RNG stream and kernel changes alter results even with identical seeds.
`seed_everything` is correctly implemented; it cannot pin behaviour across
library versions.

**Consequence for the write-up:** 14.3289 remains the number the team measured
and reported, and it is what the abstract and submission file carry. The README
quotes it as the reported result and shows the reproduction alongside, rather
than silently substituting one for the other. Pinning `torch==` to the original
version in `requirements.txt` would restore exact reproducibility if the
original version is known.
