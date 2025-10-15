from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(preds, labels):
    """
    Computes accuracy and F1-score.
    """
    if hasattr(preds, "detach"):
        preds = preds.detach().cpu().numpy()
    if hasattr(labels, "detach"):
        labels = labels.detach().cpu().numpy()
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='macro')
    
    return {'accuracy': acc, 'f1': f1}
