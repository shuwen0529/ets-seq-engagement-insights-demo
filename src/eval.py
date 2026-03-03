import numpy as np
from sklearn.metrics import roc_auc_score

def precision_at_top_quantile(y_true, y_score, q=0.2):
    n = len(y_true)
    k = int(np.ceil(n * q))
    idx = np.argsort(-y_score)[:k]
    return float(np.mean(y_true[idx]))

def evaluate_binary(y_true, y_score, q=0.2):
    auc = roc_auc_score(y_true, y_score)
    p_at = precision_at_top_quantile(y_true, y_score, q=q)
    return {"auc": float(auc), "precision_at_top_q": float(p_at)}
