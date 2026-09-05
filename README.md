# ML-CUP 2025 — Regression, and the MONK Benchmarks

Two supervised-learning studies: a 12-feature, 4-output regression problem from the
ML-CUP 2025 competition, and the three MONK classification benchmarks used as a
diagnostic warm-up. Placed 3rd of 130 teams in the competition.

Academic project — Machine Learning (project type B), MSc in Data Science and Business
Informatics, University of Pisa, a.a. 2025/2026. Coursework, done as team EnSIUMble with
Francesco Bernardini.

## Data

**ML-CUP 2025** (`regression/datasets/`). 500 labelled samples, 17 columns: an ID, 12
anonymised input features, 4 real-valued targets. Split 80/20 once into 400 training and
100 held-out test samples. A separate blind test set of 1,000 unlabelled samples is
scored by the course instructor.

The metric is Mean Euclidean Error — the mean L2 distance between the true and predicted
points in the 4-dimensional target space, always computed in the original scale after
inverse-transforming the predictions.

What makes it awkward:

- **500 samples is very little** for a 12-to-4 mapping, so every estimate carries a wide
  confidence interval. Fold-to-fold standard deviations of 1 to 2 MEE are normal here,
  which is large relative to the differences between competing models.
- **The features are almost collinear.** PCA on the training set puts 95.64% of the
  variance in the first component alone; four components reach 99%, and the Kaiser
  criterion keeps exactly one. The nominal 12 dimensions are closer to 4 real ones.
- **The features are anonymous.** No column semantics, so no domain-driven feature
  engineering is possible and nothing can be sanity-checked against physical meaning.

**MONK 1–3** (`classification/datasets/`). The standard benchmark: 6 categorical
attributes one-hot encoded to 17 binary features, 432 test samples each, and training
sets of 124, 169 and 122 samples. MONK-3 carries 5% deliberately mislabelled training
examples.

## Approach

**Baselines first.** OLS under 5-fold CV, then Lasso and Ridge alpha sweeps, then a PCA
variance analysis to explain what the regularisation results were saying.

**Non-linear classical models.** KNN over a grid of PCA components, k and weighting;
SVR over the four kernels covered in the course, wrapped in `MultiOutputRegressor`.

**Neural networks.** Seven grid searches run as external scripts
(`gridsearch_NN.py`, `gridsearch_NN2.py`) and summarised in the notebook — 2,656
configurations in total, each scored by 5-fold CV with the identical `KFold(seed=42)`
partition used everywhere else. The sequence, in order:

| # | Configs | What changed | Best CV MEE |
| ---: | ---: | :--- | ---: |
| 1 | 432 | small architectures, relu/tanh, adam/sgd/nesterov | 21.9522 |
| 2 | 648 | wider nets, input noise, dropout | 21.1798 |
| 3 | 576 | GELU, cosine LR, adamw/nadam/radam, MSE vs Huber vs MEE loss | 18.7036 |
| 4 | 120 | stronger weight decay only | 19.7491 |
| 5 | 240 | weight decay combined with dropout | 18.8746 |
| 6 | 240 | wider search over both | 18.2889 |
| 7 | 400 | up to 256x256x256, AdamW, wd to 0.1 | 18.0583 |

**Ensembling, in three stages.** First multi-seed averaging of the single best
configuration. Then a heterogeneous ensemble of 10 configurations — the top 5 from each
of the two regimes the grids kept surfacing, AdamW with heavy regularisation and Adam
with light — re-scored under the same 5-fold protocol. Then stacking: MLP, KNN and SVR
generate out-of-fold predictions across the 5 folds, and a `RidgeCV` meta-learner is
fitted once on the resulting 12 meta-features (3 models x 4 targets).

**Model assessment.** The 100 held-out samples are touched exactly once, at the end. No
hyperparameter is chosen at that stage. The meta-learner is not refitted before the test
prediction, and the MLP's retraining epoch count is the CV median scaled by the change in
training-set size rather than re-tuned.

Preprocessing discipline throughout: the loader returns raw values, and every
`StandardScaler` and PCA is fitted inside its fold. The MLP's early stopping runs on an
inner 90/10 split of the fold-training data, so the outer validation fold never
influences when training stops.

## What did not work

**Regularising the linear model was nearly pointless.** OLS gave 26.1816; the best Ridge
(alpha = 100) gave 25.5736 and the best Lasso (alpha = 0.1) gave 25.6353. About 2%, for a
full alpha sweep. That Ridge wanted such a large alpha was the useful part — it pointed
at collinearity, which the PCA analysis then confirmed. The problem was not an
overfitting linear model; it was that no linear model fits.

**Matching the loss to the evaluation metric gained nothing.** Grid 3 trained against
MSE, Huber and a differentiable MEE. MSE won. Optimising the metric directly did not beat
optimising a proxy.

**Turning up weight decay on its own made results worse.** Grid 4 pushed weight decay
higher and cut the generalisation gap from 5.99 to 3.88 — while pushing validation MEE
from 18.7036 to 19.7491. It was squashing weights the model needed, not just the ones it
did not.

**Dropout alone did nothing.** Across grids 5 and 6, every winning configuration came
back with dropout = 0. Dropout only appeared in a winner in grid 7, and only alongside
AdamW with weight decay at 0.1.

**An Optuna study failed to beat the manual grids and was dropped.** 54 trials targeted
at gap reduction, searching learning rate, weight decay and input noise around the best
known region. Best result 18.3823, against 18.0583 from the grid it was trying to
improve on. The trial log is kept at
`regression/results/grid_results_optuna_gap_reduction.csv`; it is not referenced from the
notebook, and nothing downstream uses it.

**The neural network was the weakest of the three base learners.** It absorbed 2,656 grid
configurations plus 54 Optuna trials; KNN needed a grid of a few dozen. On the held-out
test set the MLP scored 17.8870 and KNN scored 14.1284. A lazy learner that fits nothing
beat the part of the project that consumed most of the compute budget.

**The final architecture choice was not statistically meaningful, and the notebook says
so.** The winning configuration led the runner-up by 0.0096 MEE with a standard deviation
of 1.0940 across folds. The notebook prints a warning to that effect rather than
presenting the choice as settled.

**The stacking ensemble did not beat its own best base learner on the test set.** This is
the most uncomfortable result here and it is worth stating plainly. Stacking was clearly
ahead on the out-of-fold estimate — 14.3715 against KNN's 15.1070 — which is the estimate
model selection is supposed to rely on. On the 100 held-out samples the order flipped:
stacking 14.3289, KNN alone 14.1284. With 100 test samples that difference is well inside
the noise, so the honest reading is that the ensemble bought stability in
cross-validation rather than a demonstrable accuracy gain, and the test set is too small
to separate them.

**On MONK-3, four different model families stop at exactly 97.22%.** Logistic regression,
ridge classification, SVM and a tuned neural network all land on the same number. The
ceiling is the 5% label noise built into the dataset, not model capacity — further
architecture search there is wasted budget.

## Results

### ML-CUP regression

Baselines, 5-fold CV on the 400 training samples:

| Model | CV MEE |
| :--- | ---: |
| OLS | 26.1816 ± 0.6740 |
| Lasso (alpha = 0.1) | 25.6353 ± 0.8961 |
| Ridge (alpha = 100) | 25.5736 ± 0.8442 |

Final model, and the base learners inside it. OOF is the out-of-fold estimate over the
400 training samples; TS is the 100 held-out samples, scored once.

| Model | OOF MEE | Test MEE |
| :--- | ---: | ---: |
| MLP (5-seed average) | 18.9522 | 17.8870 |
| KNN (k=3, distance, PCA=6) | 15.1070 | 14.1284 |
| SVR (rbf, C=3, gamma=3, PCA=4) | 15.5959 | 15.7672 |
| Uniform average of the three | 15.2976 | 14.4049 |
| **Stacking, RidgeCV meta-learner** | **14.3715** | **14.3289** |

Against the OLS baseline of 26.1816, the final test MEE of 14.3289 is a 45% reduction.

The intermediate ensembling stages, all on the same 5-fold protocol: best single
configuration 18.06 ± 1.17, heterogeneous 10-model ensemble 17.390 ± 1.311, stacking
14.3715.

Final configuration: MLP 256x256x256, GELU, AdamW, lr 0.02, weight decay 0.05, averaged
over seeds [42, 123, 456, 789, 2024]; KNN k=3 distance-weighted on 6 PCA components; SVR
rbf with C=3, gamma=3 on 4 PCA components; RidgeCV meta-learner at alpha = 10.

### MONK classification

Test accuracy on the 432-sample test set of each benchmark.

| Model | MONK-1 | MONK-2 | MONK-3 |
| :--- | ---: | ---: | ---: |
| Logistic regression | 75.00% | 67.13% | 97.22% |
| Ridge classifier | 73.61% | 67.13% | 97.22% |
| KNN | 77.31% | 75.93% | 89.12% |
| SVM | 100% | 85.65% | 97.22% |
| Neural network | 100% | 100% | 97.22% |

The neural network needed only 2 hidden units with ReLU and Adam to solve MONK-1 and
MONK-2 exactly, where a degree-3 polynomial SVM reached 85.65% on MONK-2. On MONK-3 the
winner was tanh with a 4-2 topology at 97.22% test against 93.44% train — the only case
where test accuracy exceeds training accuracy, which is what fitting a noisy training set
correctly looks like.

## Reproducing the numbers

```bash
git clone https://github.com/iaconoalessandro/ml-cup-2025-regression.git
cd ml-cup-2025-regression
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python regression/run_pipeline.py
```

CPU is sufficient; no CUDA kernels are required. `--quick` uses 1 seed instead of 5 for a
smoke test (about 4 seconds), `--no-assets` skips figure generation. Paths are anchored
to the script, so the working directory does not matter.

Three things to know before running it:

- **`run_pipeline.py` overwrites `assets/metrics.json` and `assets/results_table.md`.**
  Those files are committed. Running the pipeline replaces them, so use
  `git checkout -- assets/` afterwards if you want the committed versions back.
- **The script reproduces the method, not the exact submitted number.** The submitted
  result is 14.3289, from `regression/Cup_Summary.ipynb`. The standalone script lands
  near it but not on it — 14.7006 in the committed `metrics.json` with 5 seeds, 14.5318
  in `--quick` mode with 1 seed. The difference comes from seed-dependent out-of-fold
  meta-features, which shift the RidgeCV alpha and the retraining epoch count. The two
  deterministic base learners, KNN at 14.1284 and SVR at 15.7672, reproduce exactly every
  time.
- **The grid searches are not re-run by the pipeline.** They took hours as external
  scripts; their summarised results are committed as CSVs under `regression/results/` and
  the notebook reads those. To re-run them, use `gridsearch_NN.py` and `gridsearch_NN2.py`
  directly.

To read rather than run, start with `regression/Cup_Summary.ipynb` — it carries the full
argument in order, with the derivations. Note that its prose is in Italian.

## Tech stack

Taken from the imports actually present in the scripts and notebooks.

| Purpose | Library |
| :--- | :--- |
| Neural networks | PyTorch (CPU) |
| Classical models, CV, PCA, scaling | scikit-learn |
| Numerics and data handling | numpy, pandas, scipy |
| Plots | matplotlib, seaborn |
| Notebooks | jupyterlab, ipykernel |

## Repository layout

```
regression/
  Cup_Summary.ipynb        the full study, in order, with derivations
  run_pipeline.py          standalone reproduction of the final stacking model
  gridsearch_NN.py         grids 1-2
  gridsearch_NN2.py        grids 3-7, plus the Optuna study
  ensembling.py            multi-seed and heterogeneous ensembling
  cup_loader.py            raw loading and the one 400/100 split
  make_assets.py           figures for this README
  results/                 committed grid-search logs, one CSV per grid
  datasets/                ML-CUP training and blind test CSVs
classification/
  Monks_Summary.ipynb      the three MONK benchmarks
  dataloader.py            one-hot encoding, shared by sklearn and PyTorch
  datasets/                monks-1/2/3 train and test splits
presentation/              project slides as delivered
assets/                    figures and the metrics the pipeline writes
```

## Authors

Alessandro Iacono (Data Science) and Francesco Bernardini (Computer Science, AI),
team EnSIUMble.
