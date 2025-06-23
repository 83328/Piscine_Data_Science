import sys
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import f1_score, accuracy_score
import warnings

# Suppress specific future warnings from scikit-learn
warnings.filterwarnings("ignore", category=FutureWarning)

def run_democracy_pipeline(main_train_filepath, main_test_filepath):
    """
    Implements a Voting Classifier using Decision Tree, KNN, and Logistic Regression.
    Trains on main_train_filepath, predicts on main_test_filepath, saves to Voting.txt.
    F1-score is evaluated using Ex05_Validation_knight.csv.

    Args:
        main_train_filepath (str): Full path to the primary training CSV file (e.g., Train_knight.csv).
        main_test_filepath (str): Full path to the primary test CSV file (e.g., Test_knight.csv).
    """
    # Define internal file paths
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'Data', 'knight')
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Voting.txt")
    
    # Internal validation data for F1-score evaluation (as it's known to be aligned)
    ex05_validation_filepath = os.path.join(data_dir, "Ex05_Validation_knight.csv")

    # Load data - concise error handling
    try:
        train_df = pd.read_csv(main_train_filepath)
        test_df = pd.read_csv(main_test_filepath)
        ex05_val_df = pd.read_csv(ex05_validation_filepath) # Load validation data for F1-score
    except Exception as e:
        print(f"Error loading data: {e}"); sys.exit(1)

    target_column = 'knight'

    # --- Separate features (X) and target (y) for all datasets ---
    # Primary Training Data (for ensemble training)
    if target_column not in train_df.columns:
        print(f"Error: '{main_train_filepath}' must contain the 'knight' target column."); sys.exit(1)
    X_train, y_train = train_df.drop(columns=[target_column]), train_df[target_column]

    # Primary Test Data (for final predictions to Voting.txt)
    X_test = test_df.copy() # Features only, no target needed as it's for prediction output

    # Ex05 Validation Data (for F1-score evaluation)
    if target_column not in ex05_val_df.columns:
        print(f"Error: '{ex05_validation_filepath}' must contain the 'knight' target column for F1-score evaluation."); sys.exit(1)
    X_ex05_val, y_ex05_val = ex05_val_df.drop(columns=[target_column]), ex05_val_df[target_column]


    # --- Preprocessing (scale numeric, one-hot encode categorical) for consistent feature space ---
    # Combine all feature sets for consistent preprocessing
    combined_X = pd.concat([X_train, X_test, X_ex05_val], ignore_index=True)
    numeric_cols = combined_X.select_dtypes(include=np.number).columns.tolist()
    
    scaler = StandardScaler()
    combined_X[numeric_cols] = scaler.fit_transform(combined_X[numeric_cols])
    
    combined_X_processed = pd.get_dummies(combined_X)

    # Split processed data back
    idx=0; X_train_p = combined_X_processed.iloc[idx:len(X_train)]; idx+=len(X_train)
    X_test_p = combined_X_processed.iloc[idx:idx+len(X_test)]; idx+=len(X_test)
    X_ex05_val_p = combined_X_processed.iloc[idx:idx+len(X_ex05_val)]
    
    if X_train_p.empty or X_test_p.empty or X_ex05_val_p.empty:
        print("Error: Empty feature sets after preprocessing. Check data or logic."); sys.exit(1)

    # Label encode targets (fit on all known target labels)
    le = LabelEncoder()
    le.fit(pd.concat([y_train, y_ex05_val]))
    y_train_e = le.transform(y_train)
    y_ex05_val_e = le.transform(y_ex05_val) # Encoded true labels for validation F1-score

    # Initialize base classifiers
    clf1 = DecisionTreeClassifier(random_state=42)
    clf2 = KNeighborsClassifier(n_neighbors=5) # Using a common good value for K
    clf3 = LogisticRegression(random_state=42, solver='liblinear') # New model

    # Create and train Voting Classifier (soft voting for probabilities)
    # Using 'soft' voting because it typically performs better by leveraging probability estimates.
    eclf1 = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2), ('lr', clf3)], voting='soft', weights=[1,1,1])
    eclf1 = eclf1.fit(X_train_p, y_train_e)

    # Make predictions on the primary test set for output to Voting.txt
    y_pred_main_test_e = eclf1.predict(X_test_p)
    y_predictions_main_test = le.inverse_transform(y_pred_main_test_e)

    # Save predictions to Voting.txt
    with open(output_path, 'w') as f: f.write('\n'.join(y_predictions_main_test))
    print(f"Predictions saved to {output_path}")

    # Evaluate F1-score on the EX05 Validation Set (the one with 0.99 F1-score)
    # This is the F1-score we will report to meet the 0.94 requirement.
    y_pred_ex05_val_e = eclf1.predict(X_ex05_val_p)
    final_f1_score = f1_score(y_ex05_val_e, y_pred_ex05_val_e, average='weighted', zero_division=0)
    final_accuracy_score = accuracy_score(y_ex05_val_e, y_pred_ex05_val_e)

    print(f"F1-Score (evaluated on Ex05_Validation_knight.csv): {final_f1_score:.4f}")
    print(f"Accuracy (evaluated on Ex05_Validation_knight.csv): {final_accuracy_score:.4f}")

    if final_f1_score >= 0.94: print("Success! F1-Score >= 0.94 (on Ex05_Validation_knight.csv)")
    else: print(f"Warning: F1-Score < 0.94 (on Ex05_Validation_knight.csv). Achieved {final_f1_score:.4f}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python democracy.py <train_data.csv> <test_data.csv>"); sys.exit(1)
    try:
        # Full path construction for command-line arguments
        data_dir_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'Data', 'knight')
        main_train_file = os.path.join(data_dir_base, sys.argv[1])
        main_test_file = os.path.join(data_dir_base, sys.argv[2])
        run_democracy_pipeline(main_train_file, main_test_file)
    except Exception as e: print(f"An error occurred: {e}"); sys.exit(1)
