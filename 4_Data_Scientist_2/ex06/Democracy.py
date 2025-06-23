import sys
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder # Only need for label encoding, not scaling/dummies
from sklearn.ensemble import VotingClassifier # Will not use directly, but for concepts
from sklearn.metrics import f1_score, accuracy_score
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

def run_voting_pipeline(tree_preds_filepath, knn_preds_filepath, lr_preds_filepath, validation_data_filepath):
    """
    Performs hard voting on predictions from three text files (Tree.txt, KNN.txt, Regression.txt).
    Outputs the ensemble predictions to Voting.txt and evaluates F1-score on the validation data.

    Args:
        tree_preds_filepath (str): Path to Decision Tree predictions (e.g., ../ex04/Tree.txt).
        knn_preds_filepath (str): Path to KNN predictions (e.g., ../ex05/KNN.txt).
        lr_preds_filepath (str): Path to Logistic Regression predictions (e.g., Regression.txt).
        validation_data_filepath (str): Path to the validation CSV (e.g., Ex05_Validation_knight.csv)
                                        which contains the true labels for evaluation.
    """
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Voting.txt")

    try:
        with open(tree_preds_filepath, 'r') as f: tree_preds = [l.strip() for l in f]
        with open(knn_preds_filepath, 'r') as f: knn_preds = [l.strip() for l in f]
        with open(lr_preds_filepath, 'r') as f: lr_preds = [l.strip() for l in f]
        
        # Load the validation data to get the true labels for evaluation
        val_df = pd.read_csv(validation_data_filepath)
        y_true_eval = val_df['knight'] # Assuming 'knight' column exists in validation data
    except Exception as e:
        print(f"Error loading prediction or validation files: {e}"); sys.exit(1)

    # Ensure all prediction lists have the same length and match the true labels
    if not (len(tree_preds) == len(knn_preds) == len(lr_preds) == len(y_true_eval)):
        print("Error: Prediction file lengths or validation data length mismatch. Cannot perform voting."); sys.exit(1)

    # Combine predictions into a DataFrame for easy voting
    preds_df = pd.DataFrame({
        'tree': tree_preds,
        'knn': knn_preds,
        'lr': lr_preds
    })

    # Label encode all possible class labels ('Jedi', 'Sith')
    le = LabelEncoder()
    all_labels = pd.concat([preds_df['tree'], preds_df['knn'], preds_df['lr'], y_true_eval]).unique()
    le.fit(all_labels)

    # Encode predictions and true labels to numerical for voting and F1-score
    preds_df_encoded = preds_df.apply(le.transform)
    y_true_eval_e = le.transform(y_true_eval)

    # Perform HARD VOTING
    ensemble_pred_e = (preds_df_encoded.sum(axis=1) >= (preds_df_encoded.shape[1] / 2)).astype(int)
    
    # Convert back to original labels
    y_predictions_ensemble = le.inverse_transform(ensemble_pred_e)

    # Save ensemble predictions to Voting.txt
    with open(output_path, 'w') as f: f.write('\n'.join(y_predictions_ensemble))

    # Evaluate F1-score of the ensemble
    final_f1 = f1_score(y_true_eval_e, ensemble_pred_e, average='weighted', zero_division=0)
    print(f"**********************************************************\n")
    print(f"Ensemble F1-Score: {final_f1:.4f}\n")
    print(f"**********************************************************\n")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python democracy.py <tree_preds.txt> <knn_preds.txt> <lr_preds.txt>"); sys.exit(1)
        print("Example: python democracy.py ../ex04/Tree.txt ../ex05/KNN.txt Regression.txt")
    
    # Define paths to prediction files based on command-line arguments
    # Assuming the script is run from the root, or paths are relative to current dir
    tree_file = sys.argv[1]
    knn_file = sys.argv[2]
    lr_file = sys.argv[3]

    # Path to the validation data file, used as the ground truth source
    data_dir_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'Data', 'knight')
    validation_data_file = os.path.join(data_dir_base, "Ex05_Validation_knight.csv")

    try:
        run_voting_pipeline(tree_file, knn_file, lr_file, validation_data_file)
    except Exception as e: print(f"An error occurred: {e}"); sys.exit(1)
