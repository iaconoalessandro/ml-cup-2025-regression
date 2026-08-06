"""
==============================================================================
make_assets.py  —  Figure publication-ready per il README (ML-CUP 2025)
==============================================================================

Genera in ../assets:

  learning_curve_best_mlp.png   curve di apprendimento (train/val MEE) della
                                configurazione MLP vincente, mediate sui 5 fold
  true_vs_predicted.png         valori veri vs predetti sui 100 sample held-out,
                                un pannello per ciascuna delle 4 dimensioni target
  error_distribution.png        distribuzione dell'errore euclideo per sample
  hyperparam_heatmap.png        heatmap lr x weight_decay della grid finale
  mee_comparison.png            confronto TR / VL / TS per i tre base learner
                                e per lo stacking
  ridgecv_alpha_curve.png       curva di tuning dell'alpha del meta-learner

Puo` girare in due modi:
  * chiamato da run_pipeline.py (riusa i risultati gia` calcolati)
  * standalone: `python make_assets.py` — genera le figure che dipendono
    solo dai CSV delle grid search
==============================================================================
"""

import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Ridge

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT  = os.path.dirname(BASE_DIR)
ASSETS_DIR = os.path.join(REPO_ROOT, 'assets')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FINAL_GRID = os.path.join(RESULTS_DIR, 'grid_results_nn2_reg_wd2_drop3_grid.csv')

sys.path.insert(0, BASE_DIR)

# Palette coerente con i plot della grid search: train blu, validation rosso.
C_TRAIN, C_VAL, C_TEST = '#1f4ea1', '#c62828', '#2e7d32'
DPI = 200

sns.set_theme(style='whitegrid', context='notebook')
plt.rcParams.update({
    'font.family'      : 'DejaVu Sans',
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'grid.alpha'       : 0.3,
    'grid.linestyle'   : '--',
    'figure.autolayout': False,
})


def _save(fig, name):
    os.makedirs(ASSETS_DIR, exist_ok=True)
    path = os.path.join(ASSETS_DIR, name)
    fig.savefig(path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"    salvato: assets/{name}")


# ==============================================================================
# 1. Learning curve della configurazione MLP vincente
# ==============================================================================

def plot_learning_curve(cfg, X_train, y_train, n_folds=5, cv_seed=42):
    """Ri-allena la config vincente sui 5 fold e media le curve MEE."""
    from gridsearch_NN2 import evaluate_config

    res = evaluate_config(X_train, y_train, cfg,
                          cv_seed=cv_seed, n_folds=n_folds, verbose=False)
    tr = np.array(res['train_curve'])
    vl = np.array(res['val_curve'])
    x = np.arange(1, len(tr) + 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(x, tr, color=C_TRAIN, lw=2.0, label='Training MEE')
    ax.plot(x, vl, color=C_VAL, lw=2.0, ls='--', label='Validation MEE')

    best_i = int(np.argmin(vl))
    ax.axvline(best_i + 1, color='#666', lw=1.0, ls=':', alpha=0.8)
    ax.scatter([best_i + 1], [vl[best_i]], color=C_VAL, s=70, zorder=5)
    ax.annotate(f"best val = {vl[best_i]:.2f}\nepoch {best_i + 1}",
                xy=(best_i + 1, vl[best_i]), xytext=(12, 18),
                textcoords='offset points', fontsize=10, color=C_VAL)

    gap = vl[-1] - tr[-1]
    ax.fill_between(x, tr, vl, color='#c62828', alpha=0.06)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MEE (original target scale)')
    ax.set_title(f"MLP learning curve — {cfg.get('_label', 'best config')}\n"
                 f"{n_folds}-fold averaged · final generalisation gap = {gap:.2f} MEE",
                 fontsize=12, fontweight='bold')
    ax.legend(frameon=True, framealpha=0.95)
    _save(fig, 'learning_curve_best_mlp.png')
    return res


# ==============================================================================
# 2. True vs Predicted sui 100 sample held-out
# ==============================================================================

def plot_true_vs_predicted(y_true, y_pred, title_suffix='Stacking · held-out test'):
    n_out = y_true.shape[1]
    fig, axes = plt.subplots(1, n_out, figsize=(4.1 * n_out, 4.3))
    axes = np.atleast_1d(axes)

    for j, ax in enumerate(axes):
        t, p = y_true[:, j], y_pred[:, j]
        lo = min(t.min(), p.min()); hi = max(t.max(), p.max())
        pad = 0.06 * (hi - lo)
        lo, hi = lo - pad, hi + pad

        ax.plot([lo, hi], [lo, hi], color='#999', lw=1.2, ls='--', zorder=1)
        ax.scatter(t, p, s=26, alpha=0.65, color=C_TEST,
                   edgecolor='white', linewidth=0.5, zorder=2)

        r = np.corrcoef(t, p)[0, 1]
        ax.set_title(f"Target {j + 1}   (r = {r:.3f})", fontsize=11)
        ax.set_xlabel('True'); ax.set_ylabel('Predicted' if j == 0 else '')
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect('equal', adjustable='box')

    fig.suptitle(f"True vs Predicted — {title_suffix}",
                 fontsize=13, fontweight='bold', y=1.02)
    fig.tight_layout()
    _save(fig, 'true_vs_predicted.png')


# ==============================================================================
# 3. Distribuzione dell'errore euclideo per sample
# ==============================================================================

def plot_error_distribution(y_true, preds_by_model):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))

    for name, p in preds_by_model.items():
        err = np.sqrt(np.sum((y_true - p) ** 2, axis=1))
        sns.kdeplot(err, ax=ax1, label=f"{name} (MEE {err.mean():.2f})",
                    lw=2.0, fill=False, clip=(0, None))

    ax1.set_xlabel('Per-sample Euclidean error')
    ax1.set_ylabel('Density')
    ax1.set_title('Error distribution — held-out test', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)

    data = [np.sqrt(np.sum((y_true - p) ** 2, axis=1))
            for p in preds_by_model.values()]
    bp = ax2.boxplot(data, labels=list(preds_by_model.keys()),
                     patch_artist=True, widths=0.55, showmeans=True,
                     meanprops=dict(marker='D', markerfacecolor=C_VAL,
                                    markeredgecolor=C_VAL, markersize=5))
    for patch in bp['boxes']:
        patch.set_facecolor('#dbe5f5'); patch.set_edgecolor(C_TRAIN)
    for med in bp['medians']:
        med.set_color(C_TRAIN); med.set_linewidth(1.8)

    ax2.set_ylabel('Per-sample Euclidean error')
    ax2.set_title('Error spread by model (◆ = mean = MEE)',
                  fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=15)
    fig.tight_layout()
    _save(fig, 'error_distribution.png')


# ==============================================================================
# 4. Heatmap della hyperparameter search
# ==============================================================================

def plot_hyperparam_heatmap(csv_path=FINAL_GRID):
    if not os.path.exists(csv_path):
        print(f"    (salto heatmap: {csv_path} assente)")
        return
    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.2))

    # (a) lr x weight_decay — media della MEE di validazione
    piv = df.pivot_table(index='weight_decay', columns='lr',
                         values='mean_mee', aggfunc='mean')
    sns.heatmap(piv, annot=True, fmt='.2f', cmap='viridis_r', ax=axes[0],
                cbar_kws={'label': 'mean CV MEE'}, linewidths=0.5,
                linecolor='white', annot_kws={'fontsize': 9})
    axes[0].set_title('Validation MEE: learning rate × weight decay',
                      fontsize=12, fontweight='bold')
    axes[0].set_xlabel('learning rate'); axes[0].set_ylabel('weight decay')

    # (b) architettura x optimizer
    piv2 = df.pivot_table(index='arch_str', columns='optimizer',
                          values='mean_mee', aggfunc='min')
    sns.heatmap(piv2, annot=True, fmt='.2f', cmap='viridis_r', ax=axes[1],
                cbar_kws={'label': 'best CV MEE'}, linewidths=0.5,
                linecolor='white', annot_kws={'fontsize': 9})
    axes[1].set_title('Best validation MEE: architecture × optimizer',
                      fontsize=12, fontweight='bold')
    axes[1].set_xlabel('optimizer'); axes[1].set_ylabel('hidden layers')

    fig.suptitle(f"Hyperparameter search — {len(df)} configurations, "
                 f"5-fold CV (seed 42)", fontsize=13, fontweight='bold', y=1.03)
    fig.tight_layout()
    _save(fig, 'hyperparam_heatmap.png')


# ==============================================================================
# 5. Confronto MEE tra i modelli
# ==============================================================================

def plot_mee_comparison(tr_mee, oof_mee, ts_mee):
    models = ['mlp', 'knn', 'svr', 'uniform', 'stacking']
    labels = ['MLP', 'KNN', 'SVR', 'Uniform avg', 'Stacking']
    tr = [tr_mee[m] for m in models]
    vl = [oof_mee.get(m, oof_mee.get('stacking_nested')) for m in models]
    ts = [ts_mee[m] for m in models]

    x = np.arange(len(models)); w = 0.26
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    b1 = ax.bar(x - w, tr, w, label='Train', color=C_TRAIN, alpha=0.9)
    b2 = ax.bar(x, vl, w, label='Validation (OOF)', color=C_VAL, alpha=0.9)
    b3 = ax.bar(x + w, ts, w, label='Test (held-out)', color=C_TEST, alpha=0.9)

    for bars in (b1, b2, b3):
        ax.bar_label(bars, fmt='%.2f', fontsize=8, padding=2)

    best_ts = min(ts[:3])
    ax.axhline(best_ts, color='#666', ls=':', lw=1.3,
               label=f'best single learner on test ({best_ts:.2f})')

    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('MEE (lower is better)')
    ax.set_title('Mean Euclidean Error by model and partition',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, ncol=2)
    ax.set_ylim(0, max(max(tr), max(vl), max(ts)) * 1.16)
    fig.tight_layout()
    _save(fig, 'mee_comparison.png')


# ==============================================================================
# 6. Curva di tuning dell'alpha del meta-learner
# ==============================================================================

def plot_alpha_curve(X_meta_sc, y_meta_sc, meta_sy, y_train, chosen_alpha):
    from gridsearch_NN2 import compute_mee

    alphas = np.logspace(-1, 7, 80)
    mees = []
    for a in alphas:
        r = Ridge(alpha=a).fit(X_meta_sc, y_meta_sc)
        mees.append(compute_mee(
            y_train, meta_sy.inverse_transform(r.predict(X_meta_sc))))

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.plot(alphas, mees, color=C_TRAIN, lw=2.2)
    r = Ridge(alpha=chosen_alpha).fit(X_meta_sc, y_meta_sc)
    chosen_mee = compute_mee(
        y_train, meta_sy.inverse_transform(r.predict(X_meta_sc)))
    ax.axvline(chosen_alpha, color=C_VAL, ls='--', lw=1.5, alpha=0.85,
               label=fr'RidgeCV $\alpha$ = {chosen_alpha:.1f}')
    ax.scatter([chosen_alpha], [chosen_mee], color=C_VAL, s=80, zorder=5,
               label=f'MEE = {chosen_mee:.4f}')

    ax.set_xscale('log')
    ax.set_xlabel(r'$\alpha$'); ax.set_ylabel('MEE on OOF meta-features')
    ax.set_title('Stacking meta-learner — alpha tuning curve',
                 fontsize=13, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    fig.tight_layout()
    _save(fig, 'ridgecv_alpha_curve.png')


# ==============================================================================
# ENTRY POINT usato da run_pipeline.py
# ==============================================================================

def build_all(X_train, y_train, y_test, oof, pred_ts, X_meta_sc, y_meta_sc,
              meta_sy, mlp_cfg, assets_dir=None):
    import json
    global ASSETS_DIR
    if assets_dir:
        ASSETS_DIR = assets_dir

    print("\n" + "=" * 74)
    print("  STAGE D — generazione delle figure")
    print("=" * 74 + "\n")

    with open(os.path.join(ASSETS_DIR, 'metrics.json')) as fp:
        m = json.load(fp)

    plot_true_vs_predicted(y_test, pred_ts['stacking'])
    plot_error_distribution(y_test, {
        'MLP': pred_ts['mlp'], 'KNN': pred_ts['knn'],
        'SVR': pred_ts['svr'], 'Stacking': pred_ts['stacking'],
    })
    plot_hyperparam_heatmap()
    plot_mee_comparison(m['train_mee'], m['oof_mee'], m['test_mee'])
    plot_alpha_curve(X_meta_sc, y_meta_sc, meta_sy, y_train, m['ridge_alpha'])

    # La learning curve richiede un re-training sui 5 fold: e` la piu` lenta,
    # quindi la facciamo per ultima.
    plot_learning_curve(mlp_cfg, X_train, y_train)


if __name__ == '__main__':
    print("Generazione delle figure che dipendono solo dai CSV delle grid search...")
    plot_hyperparam_heatmap()
    print("\nPer le figure che richiedono i modelli allenati: python run_pipeline.py")
