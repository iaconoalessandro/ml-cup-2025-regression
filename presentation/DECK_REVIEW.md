# Presentation review

Two decks were present in the working tree:

| File | Slides | Character |
|---|---|---|
| `Github.pptx` | 21 | Polished English rewrite, designed layout, consistent typography. **Retained.** |
| `School.pptx` | 18 | Original course submission. Rougher English, has the bibliography, carries the required course metadata. **Removed from the repo** (see note below). |

`Github.pptx` is the version kept for the repository, per the brief. It is
strictly better as a public artefact: better structured, better written, and it
carries the full result tables.

> **Before deleting `School.pptx` anywhere else:** it is the deck that was
> actually submitted for the course, and it is the only one containing the
> bibliography (14 references) and the required header block (author emails,
> team name, project type). Keep an archived copy outside the repo. Removing it
> here only removes it from the *public* repository.

---

## Accuracy check — `Github.pptx`

Every quantitative claim was checked against `regression/results/*.csv`, the
notebook outputs, and the raw dataset.

### ✅ Verified correct

| Slide | Claim | Check |
|---|---|---|
| 2 | Final test MEE **14.3289** | Matches notebook cell 67 |
| 2 | **−45.3%** vs OLS baseline 26.1816 | (26.1816−14.3289)/26.1816 = 45.27% ✓ |
| 5–7 | MONK accuracies (100% / 100% / 97.22%) | Consistent with `Monks_Summary.ipynb` |
| 9 | Lasso 25.6353 (α=0.1), Ridge 25.5736 (α=100) | Matches notebook |
| 10 | ">95% of variance in PC1 alone" | **Verified: 95.62%** on standardised X ✓ |
| 11 | KNN PCA=6, k=3, distance / SVR PCA=4, RBF, C=3, γ=3 | Matches `KNN_BEST` / `SVR_BEST` |
| 12 | First search winner C0169: `[32,16]` tanh adam, **21.952 ± 0.844** | Exact match to `grid_results_first_grid.csv` |
| 12 | Final search winner C0320: `[256,256,128]` gelu adamw lr 0.02 wd 0.1 do 0.1, **18.058 ± 1.168** | Exact match to `grid_results_nn2_reg_wd2_drop3_grid.csv` |
| 16 | Multi-seed table, ensemble 18.2175 ± 0.7553 | Matches notebook |
| 18 | RidgeCV α = 10.0 | Matches notebook cell 65 |
| 19 | Full TR/VL/TS table (all 12 figures) | All match notebook outputs |
| 19 | Validation-to-test drift 0.0426 | 14.3715 − 14.3289 = 0.0426 ✓ |

The deck is, on the whole, unusually well-sourced — nearly every number traces
back to a committed CSV or a notebook output.

### ❌ Errors found

**1. "A two-dimensional target" — slides 3 and 8. The target is 4-dimensional.**

Slide 3: *"Error is Mean Euclidean Error over a two-dimensional target"*
Slide 8: *"a collinear 12-feature input space, a two-dimensional target"*

The ML-CUP25 target has **4** columns (`df.iloc[:, 13:17]`), every network is
built with `output_dim=4`, and `EnSIUMble_ML-CUP25-TS.csv` carries 4 prediction
columns per row. Both slides should read "four-dimensional target".

This matters more than a typo: the MEE is a Euclidean norm *over the target
dimensions*, so stating the wrong dimensionality misstates the metric itself.

---

**2. "The meta-learner beats every member on both validation and test" — slide
19. False on test.**

Slide 19's own table, one line above the claim:

| Model | TS MEE |
|---|---|
| K-NN | **14.1284** |
| Stacking | 14.3289 |

KNN alone beats the stack by 0.20 MEE on held-out data. The stack does win on
OOF validation (14.3715 vs 15.1070), so the claim is half right.

The independent reproduction reproduces the same ordering (KNN 14.13, stack
14.70), so this is a stable property of the model rather than run-to-run noise.

**Suggested rewrite:** *"The meta-learner beats every member on out-of-fold
validation and matches the best member on test, with the lowest
validation-to-test drift of any model in the study (0.0426) — it buys stability
across partitions rather than a lower headline number."*

That is both true and a stronger argument, since it is the same "buys stability,
not a better headline" framing the deck already uses well for Stage 1 on
slide 16.

---

**3. "17.5% improvement over its best member" — slide 21. Unsupported.**

No pairing of the reported figures yields 17.5%:

| Comparison | Value |
|---|---|
| vs KNN on test (best member) | **−1.4%** (worse) |
| vs KNN on OOF (best member) | +4.9% |
| vs MLP on test | +19.9% |
| vs MLP on OOF | +24.2% |

If the intended baseline was the MLP, the honest figure is ≈20% on test —
but calling the MLP the "best member" contradicts slide 19. Recommend
recasting against the OLS baseline (−45.3%, already on slide 2 and correct) or
dropping the percentage.

---

**4. "66 epochs" — slide 18. Should be 68.**

Notebook cell 67 prints `Epoche (scaled): 68 ← usato per retraining`
(49 × 400/288 = 68.06 → 68). Minor, but the slide is otherwise precise about
hyperparameters.

> See [`../AUDIT.md`](../AUDIT.md) §C4 — the 400/288 rescaling that produces
> this number is itself methodologically questionable.

---

**5. MONK-3 Ridge hyperparameter — slide 7. Should be λ = 10.0, not 0.001.**

Slide 7 lists MONK-3's Ridge Classifier as `L2 · λ = 0.001`. The notebook
(`Monks_Summary.ipynb`, cell 61) reports `{'alpha': 10.0}` for MONK-3. `0.001`
is MONK-1's Ridge alpha (slide 5, correct there) — it appears to have been
copied down a row.

The accuracy shown alongside it (97.22%) is correct.

---

**6. `School.pptx` slide 1: "Date 15/05/2026".**

The submission file `EnSIUMble_ML-CUP25-TS.csv` and the abstract both say
**15/05/2025**. One of the two is wrong; the CSV header is the one that was
actually submitted.

---

## Summary

| Severity | Count | Items |
|---|---|---|
| Significant | 2 | Target dimensionality (×2 slides); unsupported "beats every member on test" |
| Moderate | 1 | Unsupported 17.5% improvement figure |
| Minor | 3 | MONK-3 Ridge λ; 66 vs 68 epochs; date inconsistency in `School.pptx` |

Nothing found invalidates the experimental work. The two significant items are
both *claims about* results rather than errors *in* results — the underlying
numbers on slide 19 are correct and correctly sourced.
