"""Module for evaluating metrics"""

from sklearn.metrics import accuracy_score, auc, log_loss, precision_score, recall_score, roc_curve
from sklearn.preprocessing import label_binarize
import numpy as np

def eval_metrics(y_true, y_pred, y_pred_proba):
    """
    Calcule les métriques pour une classification multiclasse.

    param y_true: Les vraies étiquettes
    param y_pred: Les prédictions du modèle
    param y_pred_proba: Les probabilités prédites par le modèle
    return: Un dictionnaire contenant les métriques calculées
    """
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # Precision, Recall
    # y_true et y_pred sont les labels réels et prédits respectivement
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')

    # Log Loss
    logloss = log_loss(y_true, y_pred_proba)

    # ROC AUC (one-vs-rest)
    n_classes = y_pred_proba.shape[1]
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    roc_auc = {}
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr, tpr)

    # Moyenne des ROC AUC
    mean_roc_auc = np.mean(list(roc_auc.values()))

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'log_loss': logloss,
        'roc_auc': roc_auc,
        'mean_roc_auc': mean_roc_auc
    }
