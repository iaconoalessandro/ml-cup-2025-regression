import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay, 
                             roc_curve, auc, 
                             precision_recall_curve, average_precision_score,
                             accuracy_score, mean_squared_error)

# --- FUNZIONI PER GRID SEARCH ---
def analyze_grid_results(grid_search):
    """
    Prende un oggetto GridSearchCV fittato e stampa una tabella pandas
    ordinata con i risultati della validazione.
    """
    print("Analisi dei risultati su Validation Set (CV):")
    results = pd.DataFrame(grid_search.cv_results_)
    
    # Seleziona colonne interessanti
    cols = ['rank_test_score', 'mean_test_score', 'std_test_score'] + \
           [c for c in results.columns if c.startswith('param_')]
    
    summary = results[cols].sort_values(by='rank_test_score')
    
    # Stampa i primi 10 risultati
    with pd.option_context('display.max_rows', 10, 'display.max_columns', None, 'display.width', 1000):
        print(summary.head(10))
    print("-" * 60)

# --- FUNZIONI DI PLOTTING (CM, ROC, PR) ---
def _get_y_score(model, X_test):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        return model.decision_function(X_test)
    else:
        raise AttributeError("Il modello non ha predict_proba o decision_function")

def plot_full_analysis(model, X_test, y_test, model_name):
    """
    Genera Confusion Matrix, ROC e Precision-Recall affiancate.
    Stampa anche Accuracy e MSE.
    """
    y_pred = model.predict(X_test)
    y_score = _get_y_score(model, X_test)
    
    # Calcolo metriche puntuali
    acc = accuracy_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    print(f"\n>>> RISULTATI TEST SET: {model_name} <<<")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"MSE:      {mse:.4f}")
    
    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Analisi: {model_name}", fontsize=16)
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['0', '1'])
    disp.plot(ax=axes[0], cmap='Blues', colorbar=False)
    axes[0].set_title("Confusion Matrix")
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    axes[1].plot([0, 1], [0, 1], color='navy', linestyle='--')
    axes[1].legend(loc="lower right")
    axes[1].set_title("ROC Curve")
    
    # 3. Precision-Recall
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    avg_prec = average_precision_score(y_test, y_score)
    axes[2].plot(recall, precision, color='green', lw=2, label=f'AP = {avg_prec:.2f}')
    axes[2].legend(loc="lower left")
    axes[2].set_title("Precision-Recall Curve")
    
    plt.tight_layout()
    plt.show()