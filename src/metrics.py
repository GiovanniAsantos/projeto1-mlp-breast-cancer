from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_report_dict(y_true, y_probs, threshold=0.5, pos_label=0):
    """Métricas de classificação binária no conjunto de teste.

    load_breast_cancer usa label 0 = maligno, 1 = benigno. precision/recall/f1_score do
    sklearn assumem pos_label=1 por padrão -- sem isso explícito, mediriam a classe
    benigna, não a maligna (a que clinicamente importa: falso negativo = câncer não
    detectado). Por isso pos_label=0.
    """
    y_pred = (y_probs >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=pos_label),
        "recall": recall_score(y_true, y_pred, pos_label=pos_label),
        "f1": f1_score(y_true, y_pred, pos_label=pos_label),
        "roc_auc": roc_auc_score(y_true, y_probs),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }
