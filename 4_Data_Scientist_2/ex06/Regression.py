import sys
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

def run_logistic_regression_pipeline(train_filepath, test_filepath):
    """
    Trains a Logistic Regression model, makes predictions on a test/validation set,
    saves predictions to 'regression.txt', and evaluates its F1-score.

    Args:
        train_filepath (str): Path to the training data CSV (e.g., Ex05_Training_knight.csv).
        test_filepath (str): Path to the test/validation data CSV (e.g., Ex05_Validation_knight.csv).
    """
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Regression.txt")

    try:
        train_df = pd.read_csv(train_filepath)
        test_df = pd.read_csv(test_filepath) # This is used for both prediction output and F1 eval
    except Exception as e:
        print(f"Error loading data: {e}"); sys.exit(1)

    target_column = 'knight'

    # Separate features (X) and target (y) for training
    if target_column not in train_df.columns:
        print(f"Error: Training file '{train_filepath}' must contain the '{target_column}' column."); sys.exit(1)
    X_train, y_train = train_df.drop(columns=[target_column]), train_df[target_column]

    # Separate features and true labels for evaluation/prediction output
    if target_column not in test_df.columns:
        print(f"Error: Test/Validation file '{test_filepath}' must contain the '{target_column}' column for evaluation."); sys.exit(1)
    X_test, y_true_eval = test_df.drop(columns=[target_column]), test_df[target_column]


    # Preprocessing (scale numeric, one-hot encode categorical)
    combined_X = pd.concat([X_train, X_test], ignore_index=True)
    numeric_cols = combined_X.select_dtypes(include=np.number).columns.tolist()
    
    scaler = StandardScaler()
    combined_X[numeric_cols] = scaler.fit_transform(combined_X[numeric_cols])
    
    combined_X_processed = pd.get_dummies(combined_X)

    # Split processed data back
    X_train_p = combined_X_processed.iloc[:len(X_train)]
    X_test_p = combined_X_processed.iloc[len(X_train):]
    
    if X_train_p.empty or X_test_p.empty:
        print("Error: Empty feature sets after preprocessing. Check data or logic."); sys.exit(1)

    # Label encode targets
    le = LabelEncoder()
    le.fit(pd.concat([y_train, y_true_eval]))
    y_train_e = le.transform(y_train)
    y_true_eval_e = le.transform(y_true_eval)

    # Initialize and train Logistic Regression model
    lr_clf = LogisticRegression(random_state=42, solver='liblinear')
    lr_clf.fit(X_train_p, y_train_e)

    # Make predictions
    y_pred_e = lr_clf.predict(X_test_p)
    y_predictions = le.inverse_transform(y_pred_e)

    # Save predictions to regression.txt
    with open(output_path, 'w') as f: f.write('\n'.join(y_predictions))

    # Evaluate F1-score
    f1 = f1_score(y_true_eval_e, y_pred_e, average='weighted', zero_division=0)
    print(f"----------------------------------------------------------\n")
    print(f"Logistic Regression F1-Score: {f1:.4f}\n")
    print(f"----------------------------------------------------------\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python logistic_regression.py <train_data.csv> <test_data.csv>"); sys.exit(1)
    try:
        data_dir_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'Data', 'knight')
        train_file = os.path.join(data_dir_base, sys.argv[1])
        test_file = os.path.join(data_dir_base, sys.argv[2])
        run_logistic_regression_pipeline(train_file, test_file)
    except Exception as e: print(f"An error occurred: {e}"); sys.exit(1)
