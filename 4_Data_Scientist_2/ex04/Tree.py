import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import f1_score
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

def train_and_predict_decision_tree(train_filepath, test_filepath, output_filepath, truth_filepath=None):
    """
    Trains a Decision Tree Classifier, makes predictions, evaluates F1-score (if truth file is provided),
    saves predictions to a file, and visualizes the trained decision tree.

    Args:
        train_filepath (str): Full path to the training CSV file (must contain 'knight' target).
        test_filepath (str): Full path to the testing CSV file (features only, no 'knight' column expected).
        output_filepath (str): Full path for the output predictions file (e.g., Tree.txt).
        truth_filepath (str, optional): Full path to the ground truth labels for the test set.
                                        If provided, F1-score will be calculated. Defaults to None.
    """
    try:
        train_df = pd.read_csv(train_filepath)
        test_df = pd.read_csv(test_filepath)
    except FileNotFoundError:
        print(f"Error: One of the data files ('{train_filepath}' or '{test_filepath}') not found.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: One of the data files ('{train_filepath}' or '{test_filepath}') is empty.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV data files: {e}")
        sys.exit(1)

    target_column = 'knight'

    # --- Prepare training data ---
    if target_column not in train_df.columns:
        print(f"Error: Training file '{train_filepath}' must contain the '{target_column}' column.")
        sys.exit(1)
    X_train = train_df.drop(columns=[target_column])
    y_train = train_df[target_column]

    # --- Prepare testing data and ground truth ---
    y_true = None
    # Check if the test file itself contains the ground truth (the target column)
    if target_column in test_df.columns:
        print(f"Found '{target_column}' column in the test file. It will be used for F1-score calculation.")
        y_true = test_df[target_column]
        X_test = test_df.drop(columns=[target_column])
    else:
        # If no target in test file, it's a blind test set.
        X_test = test_df.copy()
        # Check for an external truth file if one was provided.
        if truth_filepath:
            try:
                with open(truth_filepath, 'r') as f:
                    # Use pd.Series to handle potential index issues later
                    y_true = pd.Series([line.strip() for line in f])
            except FileNotFoundError:
                print(f"Warning: External truth file not found at {truth_filepath}. F1-score cannot be calculated.")

    # --- Align columns before any other processing ---
    train_features = X_train.columns
    # Add any columns present in train_features but not in X_test, filling with 0
    missing_in_test = set(train_features) - set(X_test.columns)
    for col in missing_in_test:
        X_test[col] = 0
    # Drop any columns present in X_test but not in train_features
    extra_in_test = set(X_test.columns) - set(train_features)
    X_test = X_test.drop(columns=list(extra_in_test))
    # Ensure the order of columns in the test set matches the training set
    X_test = X_test[train_features]


    # Handle categorical target variable ('Jedi', 'Sith')
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    class_names = label_encoder.classes_.tolist() # e.g., ['Jedi', 'Sith']

    # --- Unified Preprocessing for Features (Scaling + One-Hot Encoding) ---
    # Concatenate for unified scaling and one-hot encoding to ensure consistent columns
    combined_features = pd.concat([X_train, X_test], ignore_index=True)

    # Identify numeric columns for scaling
    numeric_cols = combined_features.select_dtypes(include=np.number).columns.tolist()

    scaler = StandardScaler()
    if numeric_cols:
        combined_features[numeric_cols] = scaler.fit_transform(combined_features[numeric_cols])

    # Apply one-hot encoding to all suitable columns (numeric and categorical)
    all_features_processed = pd.get_dummies(combined_features)

    # Split back into processed training and testing sets
    X_train_processed = all_features_processed.iloc[:len(X_train)]
    X_test_processed = all_features_processed.iloc[len(X_train):]

    if X_train_processed.empty or X_test_processed.empty:
        print("Error: No features remaining after preprocessing. Check your data or preprocessing logic.")
        sys.exit(1)

    # Initialize and train the Decision Tree Classifier
    dt_classifier = DecisionTreeClassifier(random_state=42)
    dt_classifier.fit(X_train_processed, y_train_encoded)

    # Make predictions on the test set
    y_pred_encoded = dt_classifier.predict(X_test_processed)
    y_predictions = label_encoder.inverse_transform(y_pred_encoded)

    # --- Calculate F1-score ---
    if y_true is not None:
        # Ensure y_true is a list/array to avoid index mismatch issues
        y_true_list = list(y_true)
        if len(y_true_list) != len(y_predictions):
            print(f"\n--- F1-SCORE CALCULATION SKIPPED DUE TO DATA MISMATCH ---")
            print(f"Error: Number of predictions ({len(y_predictions)}) does not match number of true labels ({len(y_true_list)}).")
            print(f"Please ensure your test data and ground truth have the same number of entries.")
            print(f"----------------------------------------------------------\n")
        else:
            # Use the original string labels for f1_score calculation
            f1 = f1_score(y_true_list, y_predictions, average='weighted', zero_division=0)
            print(f"Model F1-Score: {f1:.4f}")
    else:
        print("No ground truth available. F1-score not calculated.")

    # Save predictions to file
    try:
        with open(output_filepath, 'w') as f:
            for pred in y_predictions:
                f.write(f"{pred}\n")
        print(f"Predictions saved to: {output_filepath}")
    except Exception as e:
        print(f"Error saving predictions to file: {e}")

    # Visualize the Decision Tree
    plt.figure(figsize=(14, 8))
    plot_tree(
        dt_classifier,
        feature_names=X_train_processed.columns.tolist(),
        class_names=class_names, # Use the extracted class names directly
        filled=True,
        rounded=True,
        fontsize=8,
        max_depth=8 # Limit depth for readability, adjust as needed
    )
    plt.title("Decision Tree Classifier", fontsize=18)
    plt.show()

def main():
    # Expects 2 or 3 file arguments, which means sys.argv will have a length of 3 or 4.
    if not (3 <= len(sys.argv) <= 4):
        print("Usage: python Tree.py <train_data_filename> <test_data_filename> [truth_filename (optional)]")
        print("Example (predictions only): python Tree.py Train_knight.csv Test_knight.csv")
        print("Example (with F1-score): python Tree.py Ex05_Training_knight.csv Ex05_Validation_knight.csv")
        sys.exit(1)

    # Base directory for data files, relative to where the script is run
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'Data', 'knight')
    output_dir = os.path.dirname(os.path.abspath(__file__))

    train_filename = sys.argv[1]
    test_filename = sys.argv[2]
    output_filename = "Tree.txt" # Name of the output prediction file
    truth_filename = sys.argv[3] if len(sys.argv) == 4 else None

    train_filepath = os.path.join(data_dir, train_filename)
    test_filepath = os.path.join(data_dir, test_filename)
    output_filepath = os.path.join(output_dir, output_filename)
    truth_filepath = os.path.join(data_dir, truth_filename) if truth_filename else None

    print(f"Predictions will be saved to: {output_filepath}")
    if truth_filepath:
        print(f"Truth labels for F1-score from: {truth_filepath}\n")
    else:
        print("No truth file provided, F1-score will not be calculated.\n")

    train_and_predict_decision_tree(train_filepath, test_filepath, output_filepath, truth_filepath)

if __name__ == "__main__":
    main()
