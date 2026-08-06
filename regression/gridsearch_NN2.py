"""
==============================================================================
gridsearch_NN2.py  —  HANDMADE Grid Search v2 per la Rete Neurale (ML-CUP 2025)

DIFFERENZE RISPETTO A gridsearch_NN.py
--------------------------------------------------------------------
Mantiene tutta la struttura del precedente (CV K-fold, scaling fold-by-fold,
init activation-specific, EarlyStopping con min-delta adattivo, plot uniformi
train-blue / val-red, salvataggio CSV+JSON). Cambiano solo i punti seguenti:

[N1]  LOSS_REGISTRY: il training puo` minimizzare MSE, Huber (SmoothL1) o
      MEE differenziabile (clamp(min=1e-12) per stabilita` del gradiente in
      zero).  La metrica riportata resta sempre la MEE in scala originale.
      Motivazione: la prima e seconda grid hanno mostrato che train MEE e
      val MEE si muovono assieme (ρ ≈ 0.986), quindi la rete e` ancora in
      regime di underfitting; allinearla alla metrica di valutazione (MEE
      o Huber-1) puo` rimuovere il mismatch MSE/MEE.

[N2]  COSINE LR SCHEDULER (al posto di ReduceLROnPlateau).
      LambdaLR(epoch) = eta_min_ratio + 0.5*(1 - eta_min_ratio) *
                        (1 + cos(pi*epoch/T_max))    se epoch < T_max
                      = eta_min_ratio                altrimenti
      Cosı` lr decade smooth fino a (eta_min_ratio * lr_iniziale) e poi
      resta costante.  Niente comportamento ciclico oltre T_max.
      Permette il fine-tuning che ReduceLROnPlateau non stava facendo
      (la 2nd grid plateau-a a ~21 con lr ancora ~5e-3).

[N3]  GRID SPACE LEAN.  Sulla base dell'analisi second_grid:
       - Solo Adam   (Nesterov perdeva 1.1 MEE in media)
       - Solo ReLU   (tanh equivalente nei top, ReLU 0.6 MEE meglio in media)
       - Solo bs=32  (gli altri batch erano peggiori)
       - Dropout=0.0 fisso (effetto monotono peggiore)
       - input_noise=0.0 fisso (effetto trascurabile a σ piccoli)
       Si liberano gli assi piu` informativi:
       - 4 architetture piu` grandi (capacity)
       - 4 weight_decay (regolarizzazione vera)
       - 3 lr (range adattato all'arch piu` grande)
       - 3 loss_fn (mse, huber, mee)
       Totale: 4 × 4 × 3 × 3 = 144 config.

[N4]  TAG di esperimento default: 'nn2_grid' (cosı` non collide coi tag
      delle grid precedenti).
==============================================================================

Le note storiche [R1..R8] della versione precedente restano valide:
- R1: niente BatchNorm (Wu & He 2018, Group Normalization)
- R2: training su dati standardizzati, MEE riportata in scala originale
- R3: vedi cup_loader.py (NB: il loader continua a invocare seed_everything
      al suo interno; il commento R3 originale era una mis-documentation,
      qui non lo riportiamo come "fix gia` fatto" perche` non lo e`)
- R4: init activation-specific (Glorot/Kaiming/LeCun)
- R5: build_optimizer con inspect.signature
- R6: plot uniforme train-blue / val-red
- R7: config block in testa al file
- R8: GaussianNoise (Bishop 1995) [disponibile, non usata in questa grid]
==============================================================================
"""

import os
import json
import time
import copy
import math
import inspect
import warnings
from itertools import product

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

from cup_loader import MLCupLoader, seed_everything


# ==============================================================================
# <<<<< CONFIG BLOCK  —  MODIFICA QUI PER CAMBIARE L'ESPERIMENTO ================
# ==============================================================================

DATASET_PATH   = os.path.join('datasets', 'ML-CUP25-TR.csv')

# Tag dell'esperimento (ogni run produce file separati)
EXPERIMENT_TAG = 'nn2_reg_wd2_drop3_grid'

# Directory di output (stile nn_gridsearch.py)
RESULTS_DIR    = 'results'
PLOTS_DIR      = 'plots'

# -------- Protocollo (seed, K) --------
GLOBAL_SEED    = 42
N_FOLDS        = 5

# -------- Training --------
MAX_EPOCHS     = 500
PATIENCE       = 20
ES_REL_DELTA   = 1e-3       # miglioramento relativo minimo
ES_ABS_FLOOR   = 1e-2       # miglioramento assoluto minimo

# -------- Cosine LR Scheduler (N2) --------
# Decadimento cosine fino a eta_min_ratio * lr_iniziale, poi piatto.
COSINE_T_MAX         = 200    # epoche su cui si sviluppa il decay
COSINE_ETA_MIN_RATIO = 1e-3   # eta_min = lr_init * 1e-4 (es. 0.01 -> 1e-6)

# -------- Plot --------
PLOT_DPI       = 140
PLOT_FIGSIZE   = (8, 4.5)

# ==============================================================================
# GRID SPACE  —  il SOLO blocco da toccare per cambiare la ricerca
# ==============================================================================

# Architetture: ogni elemento e` una lista di interi (neuroni per hidden layer).
# Capacity aumentata vs second_grid (che era cap a [64, 64, 32]).
ARCHITECTURES = [
    [128, 128, 64],
    [128, 128, 128],
    [256, 128, 64],
    [256, 256, 128],
    [256, 256, 256]
]

# Grid COMUNE a tutti gli optimizer (lean, vedi N3).
COMMON_GRID = {
    'activation'  : ['gelu'],
    'lr'          : [0.005, 0.008, 0.01, 0.02],
    'weight_decay': [1e-3, 5e-3, 1e-2, 5e-2, 1e-1],
    'dropout'     : [0.0, 0.1],
    'batch_size'  : [32],
    'input_noise' : [0.0],
    'loss_fn'     : ['mse'],
}

# Grid CONDIZIONALE sugli optimizer (solo Adam, vedi N3).
OPTIMIZER_GRID = {
    'adam'   : {},
    'adamw'   : {},
    # 'nadam'   : {},
    # 'radam'   : {},
}


# ==============================================================================
# REGISTRIES (activation + optimizer + loss)
# ==============================================================================

ACTIVATION_REGISTRY = {
    'relu'       : nn.ReLU,
    'leaky_relu' : lambda: nn.LeakyReLU(0.01),
    'gelu'       : nn.GELU,
    'elu'        : nn.ELU,
    'selu'       : nn.SELU,
    'tanh'       : nn.Tanh,
    'sigmoid'    : nn.Sigmoid,
    'silu'       : nn.SiLU,
    'mish'       : nn.Mish,
    'softplus'   : nn.Softplus,
}

OPTIMIZER_REGISTRY = {
    'adam'    : optim.Adam,
    'adamw'   : optim.AdamW,
    'sgd'     : optim.SGD,
    'nesterov': optim.SGD,        # SGD + nesterov=True (vedi build_optimizer)
    'rmsprop' : optim.RMSprop,
    'nadam'   : optim.NAdam,
    'radam'   : optim.RAdam,
}


# ----- N1: LOSS REGISTRY ------------------------------------------------------

class MEELoss(nn.Module):
    """
    Mean Euclidean Error differenziabile (multi-output).

    e_p = || y_true - y_pred ||_2,  MEE = mean_p e_p

    L'argomento di sqrt e` clamp-ato a 1e-12 per evitare il gradiente
    esplosivo di sqrt(x) in x=0 quando una predizione coincide col target.
    """
    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        diff      = y_true - y_pred
        sum_sq    = torch.sum(diff ** 2, dim=1)
        distances = torch.sqrt(torch.clamp(sum_sq, min=self.eps))
        return torch.mean(distances)


LOSS_REGISTRY = {
    'mse'  : lambda: nn.MSELoss(),
    # SmoothL1Loss(beta=1.0) = Huber con delta=1.0 (in unita` di y standardizzato).
    'huber': lambda: nn.SmoothL1Loss(beta=1.0),
    'mee'  : lambda: MEELoss(),
}


# ==============================================================================
# INIZIALIZZAZIONE PESI — ACTIVATION-SPECIFIC (R4, invariato)
# ==============================================================================

def init_linear_for_activation(layer, activation, is_output=False):
    """
    Inizializza un nn.Linear in modo coerente con l'attivazione che segue.

      - ReLU, GELU, ELU, SiLU, Mish, Softplus  -> Kaiming normal (He)
                                                  nonlinearity='relu'
      - LeakyReLU                              -> Kaiming normal
                                                  a=0.01, nonl='leaky_relu'
      - SELU   -> LeCun normal (std = 1/sqrt(fan_in)) [Klambauer 2017]
      - Tanh   -> Xavier normal, gain=5/3            [Glorot-Bengio 2010]
      - Sigmoid-> Xavier normal, gain=1.0
      - Output -> Xavier normal, gain=1.0 (regressione)

    I bias sono sempre a 0.
    """
    assert isinstance(layer, nn.Linear)

    if is_output:
        nn.init.xavier_normal_(layer.weight, gain=1.0)
    elif activation in ('relu', 'gelu', 'elu', 'silu', 'mish', 'softplus'):
        nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
    elif activation == 'leaky_relu':
        nn.init.kaiming_normal_(layer.weight, a=0.01, nonlinearity='leaky_relu')
    elif activation == 'selu':
        fan_in = layer.weight.size(1)
        std = 1.0 / np.sqrt(fan_in)
        nn.init.normal_(layer.weight, mean=0.0, std=std)
    elif activation == 'tanh':
        nn.init.xavier_normal_(layer.weight,
                               gain=nn.init.calculate_gain('tanh'))  # 5/3
    elif activation == 'sigmoid':
        nn.init.xavier_normal_(layer.weight, gain=1.0)
    else:
        raise ValueError(f"Attivazione '{activation}' non gestita. "
                         f"Ammesse: {list(ACTIVATION_REGISTRY.keys())}")
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


# ==============================================================================
# UTILITIES
# ==============================================================================

def compute_mee(y_true, y_pred):
    """Mean Euclidean Error (metrica ML-CUP)."""
    if isinstance(y_true, torch.Tensor): y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor): y_pred = y_pred.detach().cpu().numpy()
    return float(np.mean(np.sqrt(np.sum((y_true - y_pred) ** 2, axis=1))))


def print_separator(title, char='=', width=78):
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def config_to_id(cfg):
    arch = 'x'.join(str(h) for h in cfg['hidden_layers'])
    parts = [
        arch, cfg['activation'], cfg['optimizer'],
        f"lr{cfg['lr']:g}", f"wd{cfg['weight_decay']:g}",
        f"do{cfg['dropout']:g}", f"bs{cfg['batch_size']}",
        f"ns{cfg.get('input_noise', 0.0):g}",
        f"loss{cfg.get('loss_fn', 'mse')}",
    ]
    for k in ('momentum', 'nesterov', 'alpha'):
        if k in cfg:
            parts.append(f"{k}{cfg[k]}")
    return '_'.join(parts)


def make_output_dirs(tag):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plot_subdir = os.path.join(PLOTS_DIR, tag)
    os.makedirs(plot_subdir, exist_ok=True)
    return RESULTS_DIR, plot_subdir


def build_configs():
    """
    Espande  ARCHITECTURES x COMMON_GRID  x  OPTIMIZER_GRID[opt].
    Gestisce automaticamente gli iperparametri condizionali degli optimizer.
    """
    common_keys = list(COMMON_GRID.keys())
    common_vals = [COMMON_GRID[k] for k in common_keys]

    configs = []
    seen = set()
    for arch in ARCHITECTURES:
        for opt_name, opt_extra in OPTIMIZER_GRID.items():
            if opt_name not in OPTIMIZER_REGISTRY:
                raise ValueError(f"Optimizer '{opt_name}' non registrato.")

            extra_keys = list(opt_extra.keys()) if opt_extra else []
            extra_vals = [opt_extra[k] for k in extra_keys] if opt_extra else [()]

            for cv in product(*common_vals):
                common_cfg = dict(zip(common_keys, cv))
                if common_cfg['activation'] not in ACTIVATION_REGISTRY:
                    raise ValueError(f"Activation '{common_cfg['activation']}' non registrata.")
                if common_cfg.get('loss_fn', 'mse') not in LOSS_REGISTRY:
                    raise ValueError(f"Loss '{common_cfg['loss_fn']}' non registrata. "
                                     f"Ammesse: {list(LOSS_REGISTRY.keys())}")

                extra_iter = product(*extra_vals) if extra_keys else [()]
                for ev in extra_iter:
                    extras = dict(zip(extra_keys, ev)) if extra_keys else {}
                    cfg = {
                        'hidden_layers': list(arch),
                        'optimizer'    : opt_name,
                        **common_cfg,
                        **extras,
                    }
                    if opt_name == 'nesterov':
                        cfg['nesterov'] = True

                    cid = config_to_id(cfg)
                    if cid in seen:
                        continue
                    seen.add(cid)
                    configs.append(cfg)
    return configs


# ==============================================================================
# MODEL (niente BatchNorm — R1)  +  GaussianNoise (R8)
# ==============================================================================

class GaussianNoise(nn.Module):
    """Rumore additivo gaussiano sugli input in training mode (Bishop 1995)."""
    def __init__(self, std=0.0):
        super().__init__()
        self.std = std

    def forward(self, x):
        if self.training and self.std > 0:
            return x + torch.randn_like(x) * self.std
        return x


class DynamicNet(nn.Module):
    """
    MLP:  [GaussianNoise?] -> [Linear -> Activation -> (Dropout?)]*n -> Linear (out)
    Niente BatchNorm (R1).  Init activation-specific (R4).
    """
    def __init__(self, input_dim, output_dim, hidden_layers,
                 activation, dropout_rate=0.0, input_noise=0.0):
        super().__init__()
        act_cls = ACTIVATION_REGISTRY.get(activation)
        if act_cls is None:
            raise ValueError(f"Attivazione '{activation}' non ammessa.")

        self.activation_name = activation
        layers = []
        if input_noise > 0:
            layers.append(GaussianNoise(std=input_noise))

        in_dim = input_dim
        hidden_linears = []
        for h in hidden_layers:
            lin = nn.Linear(in_dim, h)
            hidden_linears.append(lin)
            layers.append(lin)
            layers.append(act_cls())
            if dropout_rate > 0:
                layers.append(nn.Dropout(p=dropout_rate))
            in_dim = h

        self.output_layer = nn.Linear(in_dim, output_dim)
        layers.append(self.output_layer)
        self.net = nn.Sequential(*layers)

        # Init activation-specific per i layer interni, Xavier per l'output.
        for lin in hidden_linears:
            init_linear_for_activation(lin, activation, is_output=False)
        init_linear_for_activation(self.output_layer, activation, is_output=True)

    def forward(self, x):
        return self.net(x)


def build_optimizer(cfg, model):
    """
    Costruisce l'optimizer corretto.  Usa `inspect.signature` per filtrare
    i kwargs accettati, evitando TypeError "unexpected keyword argument"
    quando un iperparam non si applica (es. Adam + momentum).
    """
    opt_name = cfg['optimizer']
    cls = OPTIMIZER_REGISTRY.get(opt_name)
    if cls is None:
        raise ValueError(f"Optimizer '{opt_name}' non ammesso.")

    kwargs = {'lr': cfg['lr'], 'weight_decay': cfg['weight_decay']}

    sig = inspect.signature(cls.__init__)
    accepted = set(sig.parameters.keys())

    for key in ('momentum', 'nesterov', 'dampening', 'alpha',
                'eps', 'centered', 'betas'):
        if key in cfg and key in accepted:
            kwargs[key] = cfg[key]

    # SGD con nesterov=True richiede momentum > 0
    if kwargs.get('nesterov', False) and kwargs.get('momentum', 0.0) <= 0:
        raise ValueError(f"nesterov=True richiede momentum>0 (cfg={cfg})")

    return cls(model.parameters(), **kwargs)


# ----- N2: Cosine LR scheduler con plateau a eta_min ------------------------

def make_cosine_with_floor(t_max: int, eta_min_ratio: float):
    """
    Lambda per LambdaLR: cosine decay da 1.0 a eta_min_ratio in [0, t_max),
    poi resta costante a eta_min_ratio per epoch >= t_max.
    Evita il comportamento ciclico di CosineAnnealingLR oltre T_max.
    """
    def lr_lambda(epoch: int) -> float:
        if epoch >= t_max:
            return eta_min_ratio
        return eta_min_ratio + 0.5 * (1.0 - eta_min_ratio) * (
            1.0 + math.cos(math.pi * float(epoch) / float(t_max))
        )
    return lr_lambda


# ==============================================================================
# EARLY STOPPING (min-delta adattivo — R4)
# ==============================================================================

class EarlyStopping:
    """
    Early stopping su validation MEE con min-delta adattivo:
      min_delta = max(abs_floor, rel_delta * |best_mee|)
    """
    def __init__(self, patience=PATIENCE,
                 rel_delta=ES_REL_DELTA, abs_floor=ES_ABS_FLOOR):
        self.patience   = patience
        self.rel_delta  = rel_delta
        self.abs_floor  = abs_floor
        self.counter    = 0
        self.best_mee   = None
        self.early_stop = False
        self.best_state = None
        self.best_epoch = 0

    def _min_delta(self):
        if self.best_mee is None:
            return self.abs_floor
        return max(self.abs_floor, self.rel_delta * abs(self.best_mee))

    def step(self, val_mee, model, epoch):
        md = self._min_delta()
        if self.best_mee is None or val_mee < self.best_mee - md:
            self.best_mee   = val_mee
            self.best_state = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
            self.counter    = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def restore(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
        return model


def _make_loader(X_np, y_np, batch_size, gen_seed):
    ds = TensorDataset(torch.tensor(X_np, dtype=torch.float32),
                       torch.tensor(y_np, dtype=torch.float32))
    g = torch.Generator()
    g.manual_seed(int(gen_seed))
    return DataLoader(ds, batch_size=batch_size, shuffle=True, generator=g)


# ==============================================================================
# TRAINING DI UN SINGOLO FOLD
#   Loss:    cfg['loss_fn']  ∈ {'mse','huber','mee'}  (N1)
#   Metric:  MEE  (riportata in scala originale per CSV + plot)
#   LR:      cosine-with-floor  (N2)
# ==============================================================================

def train_one_fold(X_tr_raw, y_tr_raw, X_va_raw, y_va_raw,
                   cfg, model_seed):
    seed_everything(model_seed)

    # --- Scaling fit-su-training-only (zero leakage, ESL 7.10.2) ---
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    X_tr = scaler_x.fit_transform(X_tr_raw)
    X_va = scaler_x.transform(X_va_raw)
    y_tr = scaler_y.fit_transform(y_tr_raw)

    input_dim  = X_tr.shape[1]
    output_dim = y_tr_raw.shape[1]

    dl = _make_loader(X_tr, y_tr, cfg['batch_size'], gen_seed=model_seed)

    model = DynamicNet(input_dim, output_dim,
                       cfg['hidden_layers'],
                       activation=cfg['activation'],
                       dropout_rate=cfg['dropout'],
                       input_noise=cfg.get('input_noise', 0.0))

    optimizer = build_optimizer(cfg, model)

    # --- LOSS scelta dal cfg (N1) ---
    loss_name = cfg.get('loss_fn', 'mse')
    criterion = LOSS_REGISTRY[loss_name]()

    # --- LR SCHEDULER: cosine-with-floor (N2) ---
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=make_cosine_with_floor(
            t_max=COSINE_T_MAX,
            eta_min_ratio=COSINE_ETA_MIN_RATIO,
        ),
    )
    es = EarlyStopping()

    X_va_t = torch.tensor(X_va, dtype=torch.float32)
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)

    train_curve, val_curve = [], []

    for epoch in range(1, MAX_EPOCHS + 1):
        # --- Training pass (loss su y scalato) ---
        model.train()
        for xb, yb in dl:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        # --- Eval: riportiamo MEE in scala ORIGINALE (R2) ---
        model.eval()
        with torch.no_grad():
            pred_tr = scaler_y.inverse_transform(model(X_tr_t).numpy())
            pred_va = scaler_y.inverse_transform(model(X_va_t).numpy())
            tr_mee  = compute_mee(y_tr_raw, pred_tr)
            va_mee  = compute_mee(y_va_raw, pred_va)
        train_curve.append(tr_mee)
        val_curve.append(va_mee)

        # cosine scheduler: step senza argomenti, una volta per epoca
        scheduler.step()
        es.step(va_mee, model, epoch)
        if es.early_stop:
            break

    es.restore(model)
    model.eval()
    with torch.no_grad():
        pred_va = scaler_y.inverse_transform(model(X_va_t).numpy())
        pred_tr = scaler_y.inverse_transform(model(X_tr_t).numpy())
    return {
        'best_val_mee'  : compute_mee(y_va_raw, pred_va),
        'best_train_mee': compute_mee(y_tr_raw, pred_tr),
        'best_epoch'    : es.best_epoch,
        'train_curve'   : train_curve,
        'val_curve'     : val_curve,
    }


# ==============================================================================
# VALUTAZIONE DI UNA CONFIGURAZIONE (K-FOLD CV, single seed)
# ==============================================================================

def evaluate_config(X_train, y_train, cfg,
                    cv_seed=GLOBAL_SEED, n_folds=N_FOLDS,
                    verbose=True):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=cv_seed)

    fold_mees, train_mees, best_epochs = [], [], []
    all_tr_curves, all_vl_curves = [], []

    for fold_idx, (tr_idx, vl_idx) in enumerate(kf.split(X_train)):
        X_tr_raw, X_vl_raw = X_train[tr_idx], X_train[vl_idx]
        y_tr_raw, y_vl_raw = y_train[tr_idx], y_train[vl_idx]

        ms = cv_seed + fold_idx
        out = train_one_fold(X_tr_raw, y_tr_raw, X_vl_raw, y_vl_raw,
                             cfg, model_seed=ms)
        fold_mees.append(out['best_val_mee'])
        train_mees.append(out['best_train_mee'])
        best_epochs.append(out['best_epoch'])
        all_tr_curves.append(out['train_curve'])
        all_vl_curves.append(out['val_curve'])

        if verbose:
            print(f"      fold {fold_idx+1}/{n_folds}  "
                  f"tr_mee={out['best_train_mee']:.4f}  "
                  f"vl_mee={out['best_val_mee']:.4f}  "
                  f"epochs={out['best_epoch']}")

    max_len = max(len(c) for c in all_tr_curves)
    def pad(curve): return curve + [curve[-1]] * (max_len - len(curve))
    tr_mat = np.array([pad(c) for c in all_tr_curves])
    vl_mat = np.array([pad(c) for c in all_vl_curves])

    fold_mees   = np.array(fold_mees)
    train_mees  = np.array(train_mees)
    best_epochs = np.array(best_epochs)

    return {
        'config'        : cfg,
        'mean_mee'      : float(fold_mees.mean()),
        'std_mee'       : float(fold_mees.std(ddof=1)) if len(fold_mees) > 1 else 0.0,
        'mean_train'    : float(train_mees.mean()),
        'gen_gap'       : float(fold_mees.mean() - train_mees.mean()),
        'mean_epochs'   : float(best_epochs.mean()),
        'min_mee'       : float(fold_mees.min()),
        'max_mee'       : float(fold_mees.max()),
        'all_mees'      : fold_mees.tolist(),
        'train_curve'   : tr_mat.mean(axis=0).tolist(),
        'val_curve'     : vl_mat.mean(axis=0).tolist(),
    }


# ==============================================================================
# PLOT — TRAIN (BLU CONTINUA) vs VAL (ROSSA TRATTEGGIATA)  (R6)
# ==============================================================================

plt.rcParams.update({
    'font.family'       : 'DejaVu Sans',
    'font.size'         : 10,
    'axes.spines.top'   : False,
    'axes.spines.right' : False,
    'axes.grid'         : True,
    'grid.alpha'        : 0.3,
    'grid.linestyle'    : '--',
})


def plot_learning_curve(result, plot_dir, short_id):
    tr = np.array(result['train_curve'])
    vl = np.array(result['val_curve'])
    x  = np.arange(1, len(tr) + 1)

    cfg = result['config']
    fig, ax = plt.subplots(figsize=PLOT_FIGSIZE)
    ax.plot(x, tr, color='#1f4ea1', linestyle='-',  linewidth=1.7,
            label='Train MEE')
    ax.plot(x, vl, color='#c62828', linestyle='--', linewidth=1.7,
            label='Validation MEE')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('MEE')

    title = (f"[{short_id}]  arch={cfg['hidden_layers']}  "
             f"act={cfg['activation']}  opt={cfg['optimizer']}  "
             f"loss={cfg.get('loss_fn', 'mse')}\n"
             f"lr={cfg['lr']:g}  wd={cfg['weight_decay']:g}  "
             f"do={cfg['dropout']:g}  bs={cfg['batch_size']}  "
             f"ns={cfg.get('input_noise', 0.0):g}")
    for k in ('momentum', 'nesterov', 'alpha'):
        if k in cfg:
            title += f"  {k}={cfg[k]}"
    title += (f"  |  mean_val={result['mean_mee']:.3f} "
              f"+/-{result['std_mee']:.3f}")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=10, loc='upper right', framealpha=0.95)
    plt.tight_layout()
    out = os.path.join(plot_dir, f"lc__{short_id}.png")
    plt.savefig(out, dpi=PLOT_DPI)
    plt.close()


# ==============================================================================
# GRID SEARCH DRIVER
# ==============================================================================

def grid_search(X_train, y_train, configs,
                tag=EXPERIMENT_TAG, verbose=True):
    results_dir, plot_dir = make_output_dirs(tag)

    print_separator(f"HANDMADE GRID SEARCH v2 (loss∈{{mse,huber,mee}}, cosine-LR)  —  tag={tag}",
                    char='#')
    total_configs = len(configs)
    total_models  = total_configs * N_FOLDS
    est_min_lo    = total_models * 2 / 60.0
    est_min_hi    = total_models * 5 / 60.0
    print(f"  Global seed       : {GLOBAL_SEED}")
    print(f"  Configurazioni    : {total_configs}")
    print(f"  K-fold CV         : {N_FOLDS}  folds  (no restart)")
    print(f"  Totale training   : {total_models}")
    print(f"  Stima tempo       : {est_min_lo:.0f}-{est_min_hi:.0f} min")
    print(f"  Output results    : {results_dir}/grid_results_{tag}.(csv|json)")
    print(f"  Output plots      : {plot_dir}/lc__*.png")
    if total_models > 5000:
        print(f"\n  !!!  WARNING: {total_models} training e` un budget elevato.")
        print(f"       Ctrl+C entro 3 secondi per abortire; altrimenti si parte.")
        time.sleep(3)

    all_rows  = []
    json_dump = []
    t0 = time.time()

    for i, cfg in enumerate(configs):
        short_id = f"C{i+1:04d}"
        cid = config_to_id(cfg)
        print(f"\n  [{i+1:>4d}/{total_configs}]  {short_id}  {cid}")

        t_start = time.time()
        res = evaluate_config(X_train, y_train, cfg, verbose=verbose)
        elapsed = time.time() - t_start

        print(f"    -> mean={res['mean_mee']:.4f}  std={res['std_mee']:.4f}  "
              f"gen_gap={res['gen_gap']:+.4f}  "
              f"epochs~{res['mean_epochs']:.0f}  [{elapsed:.1f}s]")

        plot_learning_curve(res, plot_dir, short_id)

        row = {
            'short_id'    : short_id,
            'config_id'   : cid,
            'arch_str'    : 'x'.join(str(h) for h in cfg['hidden_layers']),
            'depth'       : len(cfg['hidden_layers']),
            'width_sum'   : sum(cfg['hidden_layers']),
            'activation'  : cfg['activation'],
            'optimizer'   : cfg['optimizer'],
            'loss_fn'     : cfg.get('loss_fn', 'mse'),
            'lr'          : cfg['lr'],
            'weight_decay': cfg['weight_decay'],
            'dropout'     : cfg['dropout'],
            'batch_size'  : cfg['batch_size'],
            'input_noise' : cfg.get('input_noise', 0.0),
            'momentum'    : cfg.get('momentum', np.nan),
            'nesterov'    : cfg.get('nesterov', np.nan),
            'alpha'       : cfg.get('alpha', np.nan),
            'mean_mee'    : res['mean_mee'],
            'std_mee'     : res['std_mee'],
            'mean_train'  : res['mean_train'],
            'gen_gap'     : res['gen_gap'],
            'mean_epochs' : res['mean_epochs'],
            'min_mee'     : res['min_mee'],
            'max_mee'     : res['max_mee'],
            'elapsed_sec' : elapsed,
        }
        all_rows.append(row)

        json_dump.append({**row, **{
            'all_mees'   : res['all_mees'],
            'train_curve': res['train_curve'],
            'val_curve'  : res['val_curve'],
        }})

    total_time = time.time() - t0
    print_separator(f"GRID SEARCH COMPLETE  —  {total_time/60:.1f} min",
                    char='#')

    df = pd.DataFrame(all_rows).sort_values('mean_mee').reset_index(drop=True)

    csv_path  = os.path.join(results_dir, f'grid_results_{tag}.csv')
    json_path = os.path.join(results_dir, f'grid_results_{tag}.json')
    df.to_csv(csv_path, index=False)
    with open(json_path, 'w') as fp:
        json.dump({
            'tag'             : tag,
            'global_seed'     : GLOBAL_SEED,
            'n_folds'         : N_FOLDS,
            'max_epochs'      : MAX_EPOCHS,
            'patience'        : PATIENCE,
            'loss_function'   : 'cfg.loss_fn  (mse|huber|mee)  ->  MEE (report/plot)',
            'lr_scheduler'    : f'CosineWithFloor(T_max={COSINE_T_MAX}, '
                                f'eta_min_ratio={COSINE_ETA_MIN_RATIO})',
            'architectures'   : [list(a) for a in ARCHITECTURES],
            'common_grid'     : COMMON_GRID,
            'optimizer_grid'  : OPTIMIZER_GRID,
            'total_configs'   : total_configs,
            'total_time_min'  : total_time / 60,
            'results'         : json_dump,
        }, fp, indent=2)
    print(f"  CSV  : {csv_path}")
    print(f"  JSON : {json_path}")
    print(f"  PLOT : {plot_dir}/  ({total_configs} learning curves)")

    print("\n  Top-10 per mean MEE (lower = better):")
    cols = ['short_id', 'arch_str', 'activation', 'optimizer', 'loss_fn',
            'lr', 'weight_decay', 'dropout', 'batch_size', 'input_noise',
            'momentum', 'mean_mee', 'std_mee', 'gen_gap']
    print(df[cols].head(10).to_string(index=False))

    return df


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == '__main__':
    seed_everything(GLOBAL_SEED)

    print("=" * 78)
    print("  ML-CUP 2025 — gridsearch_NN2.py  (loss registry + cosine LR)")
    print("=" * 78)

    loader = MLCupLoader(DATASET_PATH, test_size=0.2, seed=GLOBAL_SEED)
    X_train, X_test, y_train, y_test = loader.load_and_preprocess()

    # Il test set (20%) NON viene MAI toccato in questo file.
    print(f"\n  Train: {X_train.shape}  |  Test (held-out, NON usato): {X_test.shape}")

    configs = build_configs()
    print(f"  Grid costruita: {len(configs)} configurazioni uniche")

    df = grid_search(X_train, y_train, configs, tag=EXPERIMENT_TAG)

    print("\n" + "=" * 78)
    print("  MODEL SELECTION COMPLETE.  Ispeziona results/ e plots/.")
    print("=" * 78)
