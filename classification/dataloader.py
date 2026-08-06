# File per eseguire il Pre-Processing sui dataset MONK
# L'output è valido sia per Pytorch che Scikit
# L'idea è quella di un file unico per avere coerenza almeno nel pre-proc
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

class MonkLoader:
    def __init__(self, dataset_name='monks-1'):
        """
        Loader universale per dataset tipo MONK.
        Non hardcodiamo le categorie: l'encoder le imparerà dal Training Set.
        """
        self.dataset_name = dataset_name
        
        # Configurazione UNIVERSALE dell'Encoder:
        # - sparse_output=False: restituisce array densi (numpy), non matrici sparse.
        # - dtype=np.float32: Cruciale per PyTorch (di default usa float32 su GPU). 
        #   Numpy di base userebbe float64, che spreca memoria e richiede cast successivi.
        # - handle_unknown='ignore': Se nel test set c'è una categoria mai vista nel train,
        #   non crasha ma mette tutto a 0.
        self.encoder = OneHotEncoder(sparse_output=False, 
                                     dtype=np.float32, 
                                     handle_unknown='ignore')
        
        # Flag per sapere se l'encoder è già stato addestrato
        self.is_fitted = False

    def load_and_prepare(self, monkpath):
        """
        Caricamento raw specifico per il formato .monk
        """
        # Carichiamo il CSV gestendo gli spazi multipli come separatori
        df = pd.read_csv(monkpath, sep=" ", skipinitialspace=True, header=None)
        
        # Pulizia specifica per MONK:
        # Rimuoviamo colonne vuote create dai separatori e la colonna ID (l'ultima)
        # Struttura attesa: target, a1, a2, a3, a4, a5, a6, ID
        # Selezioniamo solo le prime 7 colonne: target + 6 features
        df = df.iloc[:, 0:7]
        
        # Rinominiamo per chiarezza, anche se per l'elaborazione è irrilevante
        df.columns = ["target", "a1", "a2", "a3", "a4", "a5", "a6"]
        
        return df

    def get_data(self, train_path, test_path):
        """
        Pipeline completa: Load -> Split -> Encode
        Ritorna Numpy Arrays pronti per Sklearn o PyTorch.
        """
        train_df = self.load_and_prepare(train_path)
        test_df = self.load_and_prepare(test_path)

        # SEPARAZIONE TARGET / FEATURES
        # Target: Lo teniamo come intero (int64) e forma (N,)
        y_train = train_df["target"].values.astype(np.int64)
        y_test = test_df["target"].values.astype(np.int64)

        # Features Raw
        X_train_raw = train_df.drop("target", axis=1)
        X_test_raw = test_df.drop("target", axis=1)
        
        # ONE-HOT ENCODING
        # Fitting SOLO sul train set (Regola d'oro del ML per evitare Data Leakage)
        X_train_onehot = self.encoder.fit_transform(X_train_raw)
        self.is_fitted = True
        
        # Transform del test set usando le regole apprese dal train
        X_test_onehot = self.encoder.transform(X_test_raw)

        # DEBUG INFO
        print(f"--- Dataset Loaded: {self.dataset_name} ---")
        print(f"X_train shape: {X_train_onehot.shape} | Type: {X_train_onehot.dtype}")
        print(f"y_train shape: {y_train.shape}        | Type: {y_train.dtype}")
        print(f"X_test shape:  {X_test_onehot.shape}")
        print(f"Features originali: {X_train_raw.shape[1]} -> OneHot Features: {X_train_onehot.shape[1]}")
        print("-" * 40)

        return X_train_onehot, y_train, X_test_onehot, y_test
"""
# Test rapido 
if __name__ == "__main__":
    loader = MonkLoader('monks-2')
    try:
        X_tr, y_tr, X_ts, y_ts = loader.get_data('./datasets/monks-2.train', './datasets/monks-2.test')
    except Exception as e:
        print(f"Errore nel loading: {e}")

Risultato Corretto:
--- Dataset Loaded: monks-1 ---
X_train shape: (124, 17) | Type: float32
y_train shape: (124,)        | Type: int64
X_test shape:  (432, 17)
Features originali: 6 -> OneHot Features: 17
----------------------------------------

Possiamo andare avanti dando per assodato che funzioni!!!!
"""