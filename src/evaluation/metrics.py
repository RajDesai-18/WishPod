from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model_with_auc(y_true, y_pred, y_pred_proba=None):
    """
    Evaluate a model's performance using Accuracy, Precision, Recall, and ROC-AUC.
    If y_pred_proba is provided, ROC-AUC will be computed.

    Parameters:
    - y_true: Ground truth labels
    - y_pred: Predicted labels (binary)
    - y_pred_proba: Predicted probabilities for the positive class (optional)

    Returns:
    - A dictionary with Accuracy, Precision, Recall, and ROC-AUC (if available)
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    roc_auc = None

    # Compute ROC-AUC if probabilities are provided
    if y_pred_proba is not None:
        roc_auc = roc_auc_score(y_true, y_pred_proba)

    # Print metrics
    print("\nEvaluation Metrics:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    if roc_auc is not None:
        print(f"ROC-AUC: {roc_auc:.2f}")

    # Return metrics as a dictionary
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc,
    }


def plot_confusion_matrix(y_true, y_pred, labels=None):
    """
    Plot a confusion matrix for model predictions.

    Parameters:
    - y_true: Ground truth labels
    - y_pred: Predicted labels
    - labels: List of class labels
    """
    cm = confusion_matrix(y_true, y_pred)
    labels = labels or ['Negative', 'Positive']

    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()
