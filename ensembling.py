import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
import copy
import os

# Importiamo dal loader personalizzato
from cup_loader import MLCupLoader, seed_everything

# ==========================================
# 1. SETUP E UTILS
# ==========================================

def compute_mee(y_true, y_pred):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    return np.mean(np.sqrt(np.sum((y_true - y_pred)**2, axis=1)))

# ==========================================
# 2. EARLY STOPPING E RETE NEURALE
# ==========================================

class EarlyStopping:
    def __init__(self, patience=40, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, current_score, model):
        score = -current_score
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = copy.deepcopy(model.state_dict())
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.counter = 0

    def restore_best_weights(self, model):
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
        return model

class DynamicNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers, dropout_rate=0.04):
        super(DynamicNet, self).__init__()
        layers = []
        in_dim = input_dim
        self.act_fn = nn.GELU()

        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(self.act_fn)
            if dropout_rate > 0:
                layers.append(nn.Dropout(p=dropout_rate))
            in_dim = h_dim
        
        layers.append(nn.Linear(in_dim, output_dim))
        self.model = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.model(x)

# ==========================================
# 3. MAIN SCRIPT: STACKING & ASSESSMENT
# ==========================================

if __name__ == "__main__":
    print("\n" + "="*50)
    print(" 🚀 AVVIO ENSEMBLING STACKING & ASSESSMENT")
    print("="*50)

    # 1. Caricamento Dati
    loader = MLCupLoader('dataset/ML-CUP25-TR.csv', test_size=0.2, seed=42)
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = loader.load_and_preprocess()

    # Scaler Globale per la X (Utile per KNN e SVR che non amano i dati raw)
    scaler_x_global = StandardScaler()
    X_train_scaled = scaler_x_global.fit_transform(X_train_raw)
    X_test_scaled = scaler_x_global.transform(X_test_raw)

    # Matrici per salvare le predizioni Out-Of-Fold (Meta-Features)
    oof_NN = np.zeros_like(y_train_raw)
    oof_KNN = np.zeros_like(y_train_raw)
    oof_SVR = np.zeros_like(y_train_raw)

    print("\n[Fase 1/3] Generazione Meta-Features (5-Fold CV)...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled)):
        print(f"  -> Processando Fold {fold_idx + 1}/5...")
        seed_everything(72)

        X_tr, X_va = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_tr, y_va = y_train_raw[train_idx], y_train_raw[val_idx]

        # ---- A. Modello K-NN ----
        knn = KNeighborsRegressor(n_neighbors=4, weights='distance')
        knn.fit(X_tr, y_tr)
        oof_KNN[val_idx] = knn.predict(X_va)

        # ---- B. Modello SVR ----
        svr = MultiOutputRegressor(SVR(kernel='rbf', C=3.0, gamma=1))
        svr.fit(X_tr, y_tr)
        oof_SVR[val_idx] = svr.predict(X_va)

        # ---- C. Modello Rete Neurale ----
        scaler_y = StandardScaler()
        y_tr_scaled = scaler_y.fit_transform(y_tr)

        train_ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32), torch.tensor(y_tr_scaled, dtype=torch.float32))
        train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)

        model = DynamicNet(12, 4, [272, 288, 144], dropout_rate=0.04)
        optimizer = optim.Adam(model.parameters(), lr=0.006111, weight_decay=0.001087)
        criterion = nn.MSELoss()
        early_stopping = EarlyStopping(patience=40, min_delta=1e-4)

        for epoch in range(1, 1000):
            model.train()
            for xb, yb in train_dl:
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_pred_scaled = model(torch.tensor(X_va, dtype=torch.float32)).numpy()
                val_pred_real = scaler_y.inverse_transform(val_pred_scaled)
                val_mee = compute_mee(y_va, val_pred_real)
            
            early_stopping(val_mee, model)
            if early_stopping.early_stop:
                break
        
        # Salviamo la predizione OOF della miglior rete di questo fold
        model = early_stopping.restore_best_weights(model)
        model.eval()
        with torch.no_grad():
            oof_pred_scaled = model(torch.tensor(X_va, dtype=torch.float32)).numpy()
            oof_NN[val_idx] = scaler_y.inverse_transform(oof_pred_scaled)

    # -------------------------------------------------------------
    # ADDESTRAMENTO DEI MODELLI FINALI SU TUTTO IL TRAIN SET
    # -------------------------------------------------------------
    print("\n[Fase 2/3] Addestramento Modelli Base e Meta-Model su TUTTO il Train Set...")
    seed_everything(42)

    # 1. K-NN Finale
    knn_final = KNeighborsRegressor(n_neighbors=4, weights='distance')
    knn_final.fit(X_train_scaled, y_train_raw)

    # 2. SVR Finale
    svr_final = MultiOutputRegressor(SVR(kernel='rbf', C=3.0, gamma=1))
    svr_final.fit(X_train_scaled, y_train_raw)

    # 3. NN Finale
    # Creiamo uno split di validazione del 10% internamente per l'Early Stopping
    X_tr_f, X_va_f, y_tr_f, y_va_f = train_test_split(X_train_scaled, y_train_raw, test_size=0.1, random_state=42)
    scaler_y_final = StandardScaler()
    y_tr_f_scaled = scaler_y_final.fit_transform(y_tr_f)

    train_ds_f = TensorDataset(torch.tensor(X_tr_f, dtype=torch.float32), torch.tensor(y_tr_f_scaled, dtype=torch.float32))
    train_dl_f = DataLoader(train_ds_f, batch_size=64, shuffle=True)

    nn_final = DynamicNet(12, 4, [256, 176, 80], dropout_rate=0.05)
    optimizer_f = optim.Adam(nn_final.parameters(), lr=0.008592, weight_decay=0.001765)
    early_stopping_f = EarlyStopping(patience=40, min_delta=1e-4)

    for epoch in range(1, 1000):
        nn_final.train()
        for xb, yb in train_dl_f:
            optimizer_f.zero_grad()
            loss = criterion(nn_final(xb), yb)
            loss.backward()
            optimizer_f.step()

        nn_final.eval()
        with torch.no_grad():
            val_pred_scaled = nn_final(torch.tensor(X_va_f, dtype=torch.float32)).numpy()
            val_pred_real = scaler_y_final.inverse_transform(val_pred_scaled)
            val_mee = compute_mee(y_va_f, val_pred_real)
        
        early_stopping_f(val_mee, nn_final)
        if early_stopping_f.early_stop:
            break
    
    nn_final = early_stopping_f.restore_best_weights(nn_final)

    # 4. Meta-Model (Ridge)
    # Concateniamo le predizioni OOF per formare il set di meta-features (3 modelli * 4 target = 12 Meta-Features)
    Z_train = np.hstack((oof_NN, oof_KNN, oof_SVR))
    
    meta_model = Ridge(alpha=1.0)
    meta_model.fit(Z_train, y_train_raw)

    # -------------------------------------------------------------
    # MODEL ASSESSMENT SUL 20% DI TEST
    # -------------------------------------------------------------
    print("\n[Fase 3/3] Model Assessment sul 20% di dati Test originali (Blind!)...")
    
    # Valutiamo prima i modelli singolarmente per poterli confrontare
    test_pred_KNN = knn_final.predict(X_test_scaled)
    test_pred_SVR = svr_final.predict(X_test_scaled)
    
    nn_final.eval()
    with torch.no_grad():
        test_pred_NN_scaled = nn_final(torch.tensor(X_test_scaled, dtype=torch.float32)).numpy()
        test_pred_NN = scaler_y_final.inverse_transform(test_pred_NN_scaled)

    mee_knn = compute_mee(y_test_raw, test_pred_KNN)
    mee_svr = compute_mee(y_test_raw, test_pred_SVR)
    mee_nn = compute_mee(y_test_raw, test_pred_NN)

    # Predizione finale dell'Ensemble
    Z_test = np.hstack((test_pred_NN, test_pred_KNN, test_pred_SVR))
    ensemble_pred = meta_model.predict(Z_test)
    
    mee_ensemble = compute_mee(y_test_raw, ensemble_pred)

    print("\n" + "="*50)
    print(" 📊 RISULTATI DEL MODEL ASSESSMENT (MEE SUL TEST SET)")
    print("="*50)
    print(f" - MEE Rete Neurale:       {mee_nn:.4f}")
    print(f" - MEE K-NN:               {mee_knn:.4f}")
    print(f" - MEE SVR:                {mee_svr:.4f}")
    print(f"\n 🏆 MEE ENSEMBLE STACKING: {mee_ensemble:.4f}")
    print("="*50)