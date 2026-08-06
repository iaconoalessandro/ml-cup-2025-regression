"""
==============================================================================
run_pipeline.py  —  Riproduzione end-to-end del modello finale ML-CUP 2025
                    (team EnSIUMble)
==============================================================================

Questo script e` la versione standalone e riproducibile della pipeline
descritta in Cup_Summary.ipynb (celle 61-69). Esegue, nell'ordine:

  STAGE A   5-fold CV out-of-fold (OOF) sui 400 sample di training
            per i 3 base learner (MLP multi-seed, KNN, SVR).
  STAGE B   Fit del meta-learner RidgeCV sulle 12 meta-feature OOF.
  STAGE C   Retraining dei base learner sui 400 sample completi e
            model assessment definitivo sui 100 sample held-out.
  STAGE D   Export di metriche (JSON + Markdown) e figure in ../assets.

PROTOCOLLO ANTI-LEAKAGE
-----------------------
  * Lo split 400/100 e` fatto una volta sola su dati grezzi (cup_loader).
    Il test set non entra in nessuno scaler, in nessuna PCA, in nessun fit.
  * Ogni StandardScaler / PCA e` fittato DENTRO il fold, sul solo
    fold-training. Mai sull'intero training set prima dello split.
  * L'early stopping dell'MLP gira su un mini-split interno 90/10 del
    fold-training. Il fold di validazione esterno non e` mai usato per
    scegliere l'epoca: resta puro per la predizione OOF.
  * Il meta-learner e` fittato UNA volta sulle OOF e NON viene ri-fittato
    prima del test.

NOTA SULLA METRICA
------------------
  La loss minimizzata dall'MLP e` MSE su target standardizzato; la MEE e`
  sempre ricalcolata in scala ORIGINALE (dopo inverse_transform) ed e`
  l'unica metrica usata per early stopping, model selection e report.

Uso:
    python run_pipeline.py              # pipeline completa
    python run_pipeline.py --quick      # 1 seed invece di 5 (smoke test)
    python run_pipeline.py --no-assets  # niente figure

==============================================================================
"""

import os
import sys
import json
import time
import argparse
import warnings

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import RidgeCV, LinearRegression

warnings.filterwarnings('ignore')

# Path ancorati al file, non alla CWD.
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(BASE_DIR)
ASSETS_DIR = os.path.join(REPO_ROOT, 'assets')
sys.path.insert(0, BASE_DIR)

from cup_loader import MLCupLoader, seed_everything          # noqa: E402
from gridsearch_NN2 import (                                  # noqa: E402
    DynamicNet, LOSS_REGISTRY, compute_mee,
    build_optimizer, make_cosine_with_floor, EarlyStopping,
)

DATASET_TR = os.path.join(BASE_DIR, 'datasets', 'ML-CUP25-TR.csv')

# ==============================================================================
# CONFIGURAZIONE — iperparametri vincenti della model selection
# ==============================================================================

GLOBAL_SEED    = 42
KF_SEED        = 42            # stesso seed usato dalle grid search
K_FOLDS        = 5
TEST_SIZE      = 0.2           # 400 train / 100 held-out test
SEEDS_MLP      = [42, 123, 456, 789, 2024]

MAX_EPOCHS     = 500
COSINE_T_MAX   = 200
COSINE_ETA_MIN = 1e-3
ES_INNER_FRAC  = 0.10          # 10% del fold-train come monitor per l'ES

# MLP: vincitore del multi-seed restart (C0397 della grid nn2_reg_wd2_drop3).
# NB: NON e` il vincitore della singola grid search (C0320, 256x256x128);
# quello era favorito da un seed fortunato. Vedi README, sezione
# "Model selection".
MLP_BEST = {
    '_label'       : '256x256x256_adamw_heavy',
    'hidden_layers': [256, 256, 256],
    'activation'   : 'gelu',
    'optimizer'    : 'adamw',
    'lr'           : 0.02,
    'weight_decay' : 0.05,
    'dropout'      : 0.0,
    'batch_size'   : 32,
    'input_noise'  : 0.0,
    'loss_fn'      : 'mse',
}

KNN_BEST = {'n_neighbors': 3, 'weights': 'distance', 'pca_components': 6}
SVR_BEST = {'kernel': 'rbf', 'C': 3.0, 'gamma': 3, 'pca_components': 4}


# ==============================================================================
# BASE LEARNER — KNN e SVR (scaler + PCA fittati sempre sul solo fold-train)
# ==============================================================================

def fit_predict_knn(X_tr, y_tr, X_query, params):
    """Pipeline KNN: scaler X -> scaler y -> PCA -> KNN.
    Ritorna (pred in-sample, pred su X_query), entrambe in scala originale."""
    sx, sy = StandardScaler(), StandardScaler()
    X_tr_sc = sx.fit_transform(X_tr)
    y_tr_sc = sy.fit_transform(y_tr)
    X_q_sc  = sx.transform(X_query)

    pca    = PCA(n_components=params['pca_components'])
    X_tr_p = pca.fit_transform(X_tr_sc)
    X_q_p  = pca.transform(X_q_sc)

    knn = KNeighborsRegressor(n_neighbors=params['n_neighbors'],
                              weights=params['weights'])
    knn.fit(X_tr_p, y_tr_sc)
    return (sy.inverse_transform(knn.predict(X_tr_p)),
            sy.inverse_transform(knn.predict(X_q_p)))


def fit_predict_svr(X_tr, y_tr, X_query, params):
    """Pipeline SVR multi-output: scaler X -> scaler y -> PCA -> SVR per target."""
    sx, sy = StandardScaler(), StandardScaler()
    X_tr_sc = sx.fit_transform(X_tr)
    y_tr_sc = sy.fit_transform(y_tr)
    X_q_sc  = sx.transform(X_query)

    pca    = PCA(n_components=params['pca_components'])
    X_tr_p = pca.fit_transform(X_tr_sc)
    X_q_p  = pca.transform(X_q_sc)

    multi = MultiOutputRegressor(SVR(kernel=params['kernel'],
                                     C=params['C'],
                                     gamma=params['gamma']))
    multi.fit(X_tr_p, y_tr_sc)
    return (sy.inverse_transform(multi.predict(X_tr_p)),
            sy.inverse_transform(multi.predict(X_q_p)))


# ==============================================================================
# BASE LEARNER — MLP
# ==============================================================================

def _build_mlp(cfg, input_dim, output_dim, seed):
    seed_everything(seed)
    model = DynamicNet(
        input_dim=input_dim, output_dim=output_dim,
        hidden_layers=cfg['hidden_layers'],
        activation=cfg['activation'],
        dropout_rate=cfg['dropout'],
        input_noise=cfg.get('input_noise', 0.0),
    )
    optimizer = build_optimizer(cfg, model)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=make_cosine_with_floor(t_max=COSINE_T_MAX,
                                         eta_min_ratio=COSINE_ETA_MIN),
    )
    criterion = LOSS_REGISTRY[cfg.get('loss_fn', 'mse')]()
    return model, optimizer, scheduler, criterion


def _loader(X_sc, y_sc, batch_size, seed):
    g = torch.Generator()
    g.manual_seed(int(seed))
    ds = TensorDataset(torch.tensor(X_sc, dtype=torch.float32),
                       torch.tensor(y_sc, dtype=torch.float32))
    return DataLoader(ds, batch_size=batch_size, shuffle=True, generator=g)


def train_mlp_with_inner_es(X_fold_tr, y_fold_tr, cfg, model_seed):
    """
    Allena un MLP facendo early stopping su un mini-split interno 90/10 del
    fold-training. Il fold di validazione esterno NON viene mai visto qui:
    e` questo che rende la predizione OOF una stima onesta.
    """
    X_in, X_es, y_in, y_es = train_test_split(
        X_fold_tr, y_fold_tr, test_size=ES_INNER_FRAC, random_state=model_seed)

    sx, sy = StandardScaler(), StandardScaler()
    X_in_sc = sx.fit_transform(X_in)
    X_es_sc = sx.transform(X_es)
    y_in_sc = sy.fit_transform(y_in)

    model, optimizer, scheduler, criterion = _build_mlp(
        cfg, X_in_sc.shape[1], y_in.shape[1], model_seed)
    dl = _loader(X_in_sc, y_in_sc, cfg['batch_size'], model_seed)
    es = EarlyStopping()
    X_es_t = torch.tensor(X_es_sc, dtype=torch.float32)

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb in dl:
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            pred_es = sy.inverse_transform(model(X_es_t).numpy())
        scheduler.step()
        es.step(compute_mee(y_es, pred_es), model, epoch)
        if es.early_stop:
            break

    es.restore(model)
    return {'model': model, 'scaler_x': sx, 'scaler_y': sy,
            'best_epoch': es.best_epoch}


def train_mlp_fixed_epochs(X_tr, y_tr, cfg, seed, n_epochs):
    """Retraining finale: nessun early stopping, numero di epoche fissato
    a priori dalla CV (vedi README, nota sull'euristica di scaling)."""
    sx, sy = StandardScaler(), StandardScaler()
    X_tr_sc = sx.fit_transform(X_tr)
    y_tr_sc = sy.fit_transform(y_tr)

    model, optimizer, scheduler, criterion = _build_mlp(
        cfg, X_tr_sc.shape[1], y_tr.shape[1], seed)
    dl = _loader(X_tr_sc, y_tr_sc, cfg['batch_size'], seed)

    for _ in range(n_epochs):
        model.train()
        for xb, yb in dl:
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    return {'model': model, 'scaler_x': sx, 'scaler_y': sy}


def mlp_predict(fitted, X_query):
    X_q_sc = fitted['scaler_x'].transform(X_query)
    fitted['model'].eval()
    with torch.no_grad():
        out = fitted['model'](torch.tensor(X_q_sc, dtype=torch.float32)).numpy()
    return fitted['scaler_y'].inverse_transform(out)


# ==============================================================================
# PIPELINE
# ==============================================================================

def run(seeds_mlp, make_assets=True):
    t_start = time.time()
    seed_everything(GLOBAL_SEED)

    # ---------------------------------------------------------------- dati --
    print("=" * 74)
    print("  ML-CUP 2025 — pipeline finale (stacking MLP + KNN + SVR / RidgeCV)")
    print("=" * 74)

    loader = MLCupLoader(DATASET_TR, test_size=TEST_SIZE, seed=GLOBAL_SEED)
    X_train, X_test, y_train, y_test = loader.load_and_preprocess()
    n_train, n_out = X_train.shape[0], y_train.shape[1]

    print(f"\n  MLP  : {MLP_BEST['_label']}  (seeds={seeds_mlp})")
    print(f"  KNN  : k={KNN_BEST['n_neighbors']}, "
          f"weights={KNN_BEST['weights']}, PCA={KNN_BEST['pca_components']}")
    print(f"  SVR  : {SVR_BEST['kernel']}, C={SVR_BEST['C']}, "
          f"gamma={SVR_BEST['gamma']}, PCA={SVR_BEST['pca_components']}")

    # ------------------------------------------------- STAGE A — OOF preds --
    print("\n" + "=" * 74)
    print(f"  STAGE A — predizioni out-of-fold ({K_FOLDS}-fold, seed={KF_SEED})")
    print("=" * 74)

    oof = {k: np.zeros((n_train, n_out)) for k in ('mlp', 'knn', 'svr')}
    best_epochs = []
    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=KF_SEED)

    for fold_idx, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
        X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
        X_val, y_val = X_train[val_idx], y_train[val_idx]

        fold_preds = []
        for seed in seeds_mlp:
            fitted = train_mlp_with_inner_es(X_tr, y_tr, MLP_BEST, seed + fold_idx)
            fold_preds.append(mlp_predict(fitted, X_val))
            best_epochs.append(fitted['best_epoch'])
        oof['mlp'][val_idx] = np.mean(fold_preds, axis=0)

        _, oof['knn'][val_idx] = fit_predict_knn(X_tr, y_tr, X_val, KNN_BEST)
        _, oof['svr'][val_idx] = fit_predict_svr(X_tr, y_tr, X_val, SVR_BEST)

        print(f"    fold {fold_idx + 1}/{K_FOLDS}  "
              f"MLP={compute_mee(y_val, oof['mlp'][val_idx]):.3f}  "
              f"KNN={compute_mee(y_val, oof['knn'][val_idx]):.3f}  "
              f"SVR={compute_mee(y_val, oof['svr'][val_idx]):.3f}")

    best_epochs = np.array(best_epochs)
    n_epochs_cv = int(np.round(np.median(best_epochs)))
    print(f"\n    best_epoch: min={best_epochs.min()}  "
          f"median={n_epochs_cv}  max={best_epochs.max()}")

    oof_mee = {k: compute_mee(y_train, v) for k, v in oof.items()}
    uniform_oof = (oof['mlp'] + oof['knn'] + oof['svr']) / 3.0
    oof_mee['uniform'] = compute_mee(y_train, uniform_oof)

    # -------------------------------------------- STAGE B — meta-learner ----
    print("\n" + "=" * 74)
    print("  STAGE B — meta-learner RidgeCV sulle meta-feature OOF")
    print("=" * 74)

    X_meta = np.hstack([oof['mlp'], oof['knn'], oof['svr']])
    meta_sx, meta_sy = StandardScaler(), StandardScaler()
    X_meta_sc = meta_sx.fit_transform(X_meta)
    y_meta_sc = meta_sy.fit_transform(y_train)

    meta = RidgeCV(alphas=np.logspace(0, 7, 50))
    meta.fit(X_meta_sc, y_meta_sc)
    print(f"\n    alpha selezionato: {meta.alpha_:.4f}")

    stack_oof_pred = meta_sy.inverse_transform(meta.predict(X_meta_sc))
    oof_mee['stacking_insample'] = compute_mee(y_train, stack_oof_pred)

    # Stima dello stack: il meta-learner viene ri-fittato in CV
    # annidata sulle stesse meta-feature, cosi` i coefficienti non sono mai
    # valutati sulle righe su cui sono stati stimati.
    nested = np.zeros_like(y_train, dtype=float)
    for tr_i, va_i in KFold(n_splits=K_FOLDS, shuffle=True,
                            random_state=KF_SEED).split(X_meta_sc):
        m = RidgeCV(alphas=np.logspace(0, 7, 50))
        m.fit(X_meta_sc[tr_i], y_meta_sc[tr_i])
        nested[va_i] = meta_sy.inverse_transform(m.predict(X_meta_sc[va_i]))
    oof_mee['stacking_nested'] = compute_mee(y_train, nested)
    print(f"    MEE stacking in-sample sulle OOF : {oof_mee['stacking_insample']:.4f}")
    print(f"    MEE stacking in CV annidata      : {oof_mee['stacking_nested']:.4f}"
          "   <- stima non distorta")

    # ------------------------------- STAGE C — retraining + assessment ------
    # Le epoche osservate in CV valgono per un training da 288 sample.
    # Il retraining gira su 400 sample con lo stesso batch_size.
    n_inner = int(round(n_train * (1 - 1 / K_FOLDS) * (1 - ES_INNER_FRAC)))  # 288
    n_epochs_final = int(round(n_epochs_cv * n_train / n_inner))

    print("\n" + "=" * 74)
    print("  STAGE C — retraining su 400 sample + assessment sui 100 held-out")
    print("=" * 74)
    print(f"\n    epoche CV (mediana su {n_inner} sample) : {n_epochs_cv}")
    print(f"    epoche retraining (x {n_train}/{n_inner})     : {n_epochs_final}")

    mlp_tr_preds, mlp_ts_preds = [], []
    for seed in seeds_mlp:
        fitted = train_mlp_fixed_epochs(X_train, y_train, MLP_BEST,
                                        seed, n_epochs_final)
        mlp_tr_preds.append(mlp_predict(fitted, X_train))
        mlp_ts_preds.append(mlp_predict(fitted, X_test))

    pred_tr = {'mlp': np.mean(mlp_tr_preds, axis=0)}
    pred_ts = {'mlp': np.mean(mlp_ts_preds, axis=0)}
    pred_tr['knn'], pred_ts['knn'] = fit_predict_knn(X_train, y_train, X_test, KNN_BEST)
    pred_tr['svr'], pred_ts['svr'] = fit_predict_svr(X_train, y_train, X_test, SVR_BEST)

    # Meta-learner NON ri-fittato: usa i coefficienti stimati sulle OOF.
    X_meta_ts = np.hstack([pred_ts['mlp'], pred_ts['knn'], pred_ts['svr']])
    pred_ts['stacking'] = meta_sy.inverse_transform(
        meta.predict(meta_sx.transform(X_meta_ts)))
    X_meta_tr = np.hstack([pred_tr['mlp'], pred_tr['knn'], pred_tr['svr']])
    pred_tr['stacking'] = meta_sy.inverse_transform(
        meta.predict(meta_sx.transform(X_meta_tr)))

    pred_ts['uniform'] = (pred_ts['mlp'] + pred_ts['knn'] + pred_ts['svr']) / 3.0
    pred_tr['uniform'] = (pred_tr['mlp'] + pred_tr['knn'] + pred_tr['svr']) / 3.0

    tr_mee = {k: compute_mee(y_train, v) for k, v in pred_tr.items()}
    ts_mee = {k: compute_mee(y_test, v) for k, v in pred_ts.items()}

    # ------------------------------------------------------- report --------
    rows = [
        ('MLP (multi-seed)',   tr_mee['mlp'],      oof_mee['mlp'],               ts_mee['mlp']),
        ('KNN',                tr_mee['knn'],      oof_mee['knn'],               ts_mee['knn']),
        ('SVR',                tr_mee['svr'],      oof_mee['svr'],               ts_mee['svr']),
        ('Uniform average',    tr_mee['uniform'],  oof_mee['uniform'],           ts_mee['uniform']),
        ('Stacking (RidgeCV)', tr_mee['stacking'], oof_mee['stacking_nested'],   ts_mee['stacking']),
    ]

    print("\n" + "=" * 74)
    print("  RISULTATI FINALI — MEE (scala originale del target)")
    print("=" * 74)
    print(f"\n    {'Model':<20} {'TR':>10} {'VL (OOF)':>12} {'TS':>10}")
    print(f"    {'-' * 54}")
    for name, tr, vl, ts in rows:
        print(f"    {name:<20} {tr:>10.4f} {vl:>12.4f} {ts:>10.4f}")

    best_member_ts = min(ts_mee['mlp'], ts_mee['knn'], ts_mee['svr'])
    print(f"\n    Miglior base learner sul test : {best_member_ts:.4f}")
    print(f"    Stacking sul test             : {ts_mee['stacking']:.4f}"
          f"   (delta {ts_mee['stacking'] - best_member_ts:+.4f})")

    metrics = {
        'mlp_config'          : MLP_BEST,
        'knn_config'          : KNN_BEST,
        'svr_config'          : SVR_BEST,
        'seeds_mlp'           : list(seeds_mlp),
        'k_folds'             : K_FOLDS,
        'kf_seed'             : KF_SEED,
        'ridge_alpha'         : float(meta.alpha_),
        'n_epochs_cv_median'  : n_epochs_cv,
        'n_epochs_final'      : n_epochs_final,
        'train_mee'           : {k: float(v) for k, v in tr_mee.items()},
        'oof_mee'             : {k: float(v) for k, v in oof_mee.items()},
        'test_mee'            : {k: float(v) for k, v in ts_mee.items()},
        'best_base_learner_ts': float(best_member_ts),
        'runtime_sec'         : round(time.time() - t_start, 1),
    }

    os.makedirs(ASSETS_DIR, exist_ok=True)
    with open(os.path.join(ASSETS_DIR, 'metrics.json'), 'w') as fp:
        json.dump(metrics, fp, indent=2)
    _write_markdown_table(rows, metrics)

    if make_assets:
        from make_assets import build_all
        build_all(X_train, y_train, y_test, oof, pred_ts, X_meta_sc,
                  y_meta_sc, meta_sy, MLP_BEST, ASSETS_DIR)

    print(f"\n  Metriche  : {os.path.join(ASSETS_DIR, 'metrics.json')}")
    print(f"  Tabella   : {os.path.join(ASSETS_DIR, 'results_table.md')}")
    print(f"  Runtime   : {metrics['runtime_sec']:.1f}s")
    return metrics


def _write_markdown_table(rows, metrics):
    lines = [
        "| Model | TR MEE | VL MEE (OOF) | TS MEE |",
        "|---|---:|---:|---:|",
    ]
    for name, tr, vl, ts in rows:
        bold = name.startswith('Stacking')
        fmt = (lambda v: f"**{v:.4f}**") if bold else (lambda v: f"{v:.4f}")
        label = f"**{name}**" if bold else name
        lines.append(f"| {label} | {fmt(tr)} | {fmt(vl)} | {fmt(ts)} |")
    out = os.path.join(ASSETS_DIR, 'results_table.md')
    with open(out, 'w') as fp:
        fp.write("\n".join(lines) + "\n")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--quick', action='store_true',
                    help='usa 1 solo seed per l\'MLP (smoke test veloce)')
    ap.add_argument('--no-assets', action='store_true',
                    help='non generare le figure')
    args = ap.parse_args()

    run(seeds_mlp=SEEDS_MLP[:1] if args.quick else SEEDS_MLP,
        make_assets=not args.no_assets)
