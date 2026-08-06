import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import random
import os

def seed_everything(seed=42):
    """
    Fissa l'entropia dell'intero ambiente per garantire una 
    riproducibilità assoluta in ogni libreria utilizzata.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class MLCupLoader:
    def __init__(self, filepath, test_size=0.2, seed=42):
        self.filepath = filepath
        self.test_size = test_size
        self.seed = seed
        
        # Variabili di stato interne (ora contengono dati grezzi/non scalati)
        self.X_train_raw = None
        self.X_test_raw = None
        self.y_train_raw = None
        self.y_test_raw = None

    def load_and_preprocess(self):
        print(f"--- Inizializzazione Loader ML-CUP (Seed: {self.seed}) ---")
        seed_everything(self.seed)
        
        # 1. Lettura del CSV (ignoriamo eventuali commenti all'inizio del file)
        # Assumiamo che i dati non abbiano un header nominale (header=None)
        try:
            df = pd.read_csv(self.filepath, comment='#', header=None)
        except Exception as e:
            raise FileNotFoundError(f"Errore nella lettura del file: {e}")
            
        print(f"Dataset originale caricato. Shape: {df.shape} (Righe, Colonne)")
        
        # 2. Slicing: Rimuoviamo l'ID (Colonna 0)
        # Features (X): Colonne da 1 a 12
        # Targets (Y): Colonne da 13 a 16
        X_raw = df.iloc[:, 1:13].values
        y_raw = df.iloc[:, 13:17].values
        
        # 3. Lo Split Sacro (80/20) - SENZA SCALING
        self.X_train_raw, self.X_test_raw, self.y_train_raw, self.y_test_raw = train_test_split(
            X_raw, y_raw, 
            test_size=self.test_size, 
            random_state=self.seed
        )
        print(f"Split completato: Train Set ({len(self.X_train_raw)}), Test Set ({len(self.X_test_raw)})")
        print("I dati restituiti NON sono standardizzati (Raw Values).")
        
        return self.X_train_raw, self.X_test_raw, self.y_train_raw, self.y_test_raw

    def get_tensors(self):
        """Restituisce i dati pronti per il motore di PyTorch (formato Raw)."""
        if self.X_train_raw is None:
            raise ValueError("Devi prima chiamare load_and_preprocess()!")
            
        X_tr_t = torch.tensor(self.X_train_raw, dtype=torch.float32)
        X_ts_t = torch.tensor(self.X_test_raw, dtype=torch.float32)
        y_tr_t = torch.tensor(self.y_train_raw, dtype=torch.float32)
        y_ts_t = torch.tensor(self.y_test_raw, dtype=torch.float32)
        
        return X_tr_t, X_ts_t, y_tr_t, y_ts_t