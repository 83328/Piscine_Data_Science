import sys
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def calculate_confusion_matrix_and_metrics(y_true, y_pred, positive_class='Jedi', negative_class='Sith'):
    """
    Calculates the confusion matrix, precision, recall, f1-score, and accuracy.
    Assumes binary classification with 'Jedi' and 'Sith' labels.
    'Jedi' is considered the positive class for the first set of metrics.
    'Sith' is considered the positive class for the second set of metrics.
    """
    tp_jedi = 0  # True Positives for Jedi (Actual Jedi, Predicted Jedi)
    tn_jedi = 0  # True Negatives for Jedi (Actual Sith, Predicted Sith)
    fp_jedi = 0  # False Positives for Jedi (Actual Sith, Predicted Jedi)
    fn_jedi = 0  # False Negatives for Jedi (Actual Jedi, Predicted Sith)

    tp_sith = 0  # True Positives for Sith (Actual Sith, Predicted Sith)
    tn_sith = 0  # True Negatives for Sith (Actual Jedi, Predicted Jedi)
    fp_sith = 0  # False Positives for Sith (Actual Jedi, Predicted Sith)
    fn_sith = 0  # False Negatives for Sith (Actual Sith, Predicted Jedi)

    for i in range(len(y_true)):
        true_label = y_true[i]
        predicted_label = y_pred[i]

        # --- Metrics for Jedi as positive class ---
        if true_label == positive_class and predicted_label == positive_class:
            tp_jedi += 1
        elif true_label == negative_class and predicted_label == negative_class:
            tn_jedi += 1
        elif true_label == negative_class and predicted_label == positive_class:
            fp_jedi += 1
        elif true_label == positive_class and predicted_label == negative_class:
            fn_jedi += 1

        # --- Metrics for Sith as positive class ---
        if true_label == negative_class and predicted_label == negative_class:
            tp_sith += 1
        elif true_label == positive_class and predicted_label == positive_class:
            tn_sith += 1
        elif true_label == positive_class and predicted_label == negative_class:
            fp_sith += 1
        elif true_label == negative_class and predicted_label == positive_class:
            fn_sith += 1

    # Corrected Confusion matrix structure to match the subject's example:
    # [[TP_Jedi, FN_Jedi],
    #  [FP_Jedi, TN_Jedi]]
    # Where:
    # TP_Jedi = Actual Jedi, Predicted Jedi
    # FN_Jedi = Actual Jedi, Predicted Sith
    # FP_Jedi = Actual Sith, Predicted Jedi
    # TN_Jedi = Actual Sith, Predicted Sith
    confusion_matrix_display = np.array([[tp_jedi, fn_jedi], [fp_jedi, tn_jedi]])


    # Calculate metrics for Jedi (positive class)
    precision_jedi = tp_jedi / (tp_jedi + fp_jedi) if (tp_jedi + fp_jedi) > 0 else 0
    recall_jedi = tp_jedi / (tp_jedi + fn_jedi) if (tp_jedi + fn_jedi) > 0 else 0
    f1_score_jedi = (2 * precision_jedi * recall_jedi) / (precision_jedi + recall_jedi) if (precision_jedi + recall_jedi) > 0 else 0

    # Calculate metrics for Sith (positive class)
    precision_sith = tp_sith / (tp_sith + fp_sith) if (tp_sith + fp_sith) > 0 else 0
    recall_sith = tp_sith / (tp_sith + fn_sith) if (tp_sith + fn_sith) > 0 else 0
    f1_score_sith = (2 * precision_sith * recall_sith) / (precision_sith + recall_sith) if (precision_sith + recall_sith) > 0 else 0

    total_samples = len(y_true)
    accuracy = (tp_jedi + tn_jedi) / total_samples if total_samples > 0 else 0

    total_jedi = tp_jedi + fn_jedi
    total_sith = tp_sith + fn_sith

    return {
        'confusion_matrix': confusion_matrix_display,
        'precision_jedi': precision_jedi,
        'recall_jedi': recall_jedi,
        'f1_score_jedi': f1_score_jedi,
        'total_jedi': total_jedi,
        'precision_sith': precision_sith,
        'recall_sith': recall_sith,
        'f1_score_sith': f1_score_sith,
        'total_sith': total_sith,
        'accuracy': accuracy,
        'overall_total': total_samples
    }

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
    # Assuming ex00 is one level below the directory containing 'Data'
    # So, from ex00/, we go up one (..) to project_root/, then into Data/knight/
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

    metrics = calculate_confusion_matrix_and_metrics(truths, predictions)

    # Print output as specified in the subject
    print("           precision    recall    f1-score   total")
    print(f"Jedi         {metrics['precision_jedi']:.2f}       {metrics['recall_jedi']:.2f}        {metrics['f1_score_jedi']:.2f}     {metrics['total_jedi']}")
    print(f"Sith         {metrics['precision_sith']:.2f}       {metrics['recall_sith']:.2f}        {metrics['f1_score_sith']:.2f}     {metrics['total_sith']}")
    print(f"\naccuracy                                {metrics['accuracy']:.2f}       {metrics['overall_total']}")
    print(f"\n{metrics['confusion_matrix']}")

    # Display the confusion matrix plot
    display_confusion_matrix_plot(metrics['confusion_matrix'], ['Jedi', 'Sith'])

if __name__ == "__main__":
    main()
