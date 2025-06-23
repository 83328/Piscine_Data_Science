import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# Import the necessary functions from scikit-learn
from sklearn.metrics import confusion_matrix, classification_report

def calculate_metrics_from_sk(y_true, y_pred, labels=['Jedi', 'Sith']):
    """
    Calculates the confusion matrix and performance metrics using scikit-learn.
    This is a much simpler and more robust way than manual calculation.
    """
    # Get the confusion matrix from sklearn.
    # The `labels` parameter ensures the matrix is in the desired order.
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Get the full classification report as a dictionary.
    # This automatically calculates precision, recall, f1-score, and support for each class.
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)

    # Extract metrics into the format the main function expects, for consistent printing.
    metrics = {
        'confusion_matrix': cm,
        'precision_jedi': report['Jedi']['precision'],
        'recall_jedi':    report['Jedi']['recall'],
        'f1_score_jedi':  report['Jedi']['f1-score'],
        'total_jedi':     report['Jedi']['support'],
        'precision_sith': report['Sith']['precision'],
        'recall_sith':    report['Sith']['recall'],
        'f1_score_sith':  report['Sith']['f1-score'],
        'total_sith':     report['Sith']['support'],
        'accuracy':       report['accuracy'],
        'overall_total':  len(y_true)
    }
    return metrics

def display_confusion_matrix_plot(cm, labels, title="Confusion Matrix"):
    """
    Displays the confusion matrix as a heatmap with a 'viridis' colormap.
    """
    plt.figure(figsize=(8, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', xticklabels=labels, yticklabels=labels, annot_kws={"size": 16})
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.show()

def main():
    if len(sys.argv) != 3:
        print("Usage: python Confusion_Matrix.py <predictions_filename> <truth_filename>")
        print("Example: python Confusion_Matrix.py predictions.txt truth.txt")
        sys.exit(1)

    # Base directory for data files, relative to where the script is run
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'Data', 'knight')

    predictions_filename = sys.argv[1]
    truth_filename = sys.argv[2]

    predictions_path = os.path.join(data_dir, predictions_filename)
    truth_path = os.path.join(data_dir, truth_filename)

    try:
        with open(predictions_path, 'r') as f:
            predictions = [line.strip() for line in f]
        with open(truth_path, 'r') as f:
            truths = [line.strip() for line in f]
    except FileNotFoundError:
        print(f"Error: One of the files '{predictions_path}' or '{truth_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading files: {e}")
        sys.exit(1)

    if len(predictions) != len(truths):
        print("Error: The number of predictions and truth values do not match.")
        sys.exit(1)

    # This one line replaces the entire manual calculation.
    metrics = calculate_metrics_from_sk(truths, predictions)

    print("           precision    recall    f1-score   total")
    print(f"Jedi         {metrics['precision_jedi']:.2f}       {metrics['recall_jedi']:.2f}        {metrics['f1_score_jedi']:.2f}     {metrics['total_jedi']}")
    print(f"Sith         {metrics['precision_sith']:.2f}       {metrics['recall_sith']:.2f}        {metrics['f1_score_sith']:.2f}     {metrics['total_sith']}")
    print(f"\naccuracy                                {metrics['accuracy']:.2f}       {metrics['overall_total']}")
    print(f"\n{metrics['confusion_matrix']}")

    display_confusion_matrix_plot(metrics['confusion_matrix'], ['Jedi', 'Sith'])

if __name__ == "__main__":
    main()
