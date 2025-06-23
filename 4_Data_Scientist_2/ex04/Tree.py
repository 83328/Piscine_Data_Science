import sys, os, pandas as pd, numpy as np, matplotlib.pyplot as plt
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import f1_score; warnings.filterwarnings("ignore", category=FutureWarning)

def plot_decision_tree(model, features, classes):
    plt.figure(figsize=(14,8)); plot_tree(model, feature_names=features, class_names=classes, filled=True, rounded=True, fontsize=8, max_depth=8)
    plt.title("Decision Tree Classifier", fontsize=18); plt.show()

def run_tree_pipeline(train_path, test_path): # test_path will be Ex05_Validation_knight.csv
    # Define internal file paths
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Tree.txt")

    # Load data
    try:
        train_df = pd.read_csv(train_path)
        # For evaluation, we will use the 'test_path' file itself (Ex05_Validation_knight.csv)
        # and assume its 'knight' column is the truth.
        eval_df = pd.read_csv(test_path)
    except Exception as e: print(f"Error loading data: {e}"); sys.exit(1)

    target_column = 'knight'

    # --- Split data ---
    if target_column not in train_df.columns:
        print(f"Error: Training file '{train_path}' must contain the '{target_column}' column."); sys.exit(1)
    X_train, y_train = train_df.drop(columns=[target_column]), train_df[target_column]

    if target_column not in eval_df.columns:
        print(f"Error: Evaluation file '{test_path}' must contain the '{target_column}' column for F1-score evaluation."); sys.exit(1)
    X_eval, y_eval_true = eval_df.drop(columns=[target_column]), eval_df[target_column]

    # --- Preprocessing ---
    # Combine training and evaluation features for consistent scaling and one-hot encoding
    combined_X = pd.concat([X_train, X_eval], ignore_index=True)
    numeric_cols = combined_X.select_dtypes(include=np.number).columns.tolist()
    scaler = StandardScaler(); combined_X[numeric_cols] = scaler.fit_transform(combined_X[numeric_cols])
    combined_X_processed = pd.get_dummies(combined_X)

    # Split processed data back
    X_train_p = combined_X_processed.iloc[:len(X_train)]
    X_eval_p = combined_X_processed.iloc[len(X_train):]

    # Label encode targets
    # Fit label encoder on both training and evaluation true labels to handle all classes
    le = LabelEncoder(); le.fit(pd.concat([y_train, y_eval_true]))
    y_train_e = le.transform(y_train)
    y_eval_true_e = le.transform(y_eval_true) # Encoded true labels for evaluation

    # --- Key Model Part (Decision Tree) ---
    dt_clf = DecisionTreeClassifier(random_state=42); dt_clf.fit(X_train_p, y_train_e)
    
    # Make predictions on the evaluation set (X_eval_p)
    y_pred_e = dt_clf.predict(X_eval_p)
    y_preds = le.inverse_transform(y_pred_e)

    # Save predictions to Tree.txt (these will be predictions for the EVALUATION set)
    with open(output_path, 'w') as f: f.write('\n'.join(y_preds))
    print(f"Predictions saved to {output_path}")

    # Evaluate F1-score on the evaluation set (Ex05_Validation_knight.csv)
    f1 = f1_score(y_eval_true_e, y_pred_e, average='weighted', zero_division=0)
    print(f"----------------------------------------------------------\n")
    print(f"Model F1-Score (on {os.path.basename(test_path)}): {f1:.4f}\n")
    print(f"----------------------------------------------------------\n")

    plot_decision_tree(dt_clf, X_train_p.columns.tolist(), le.classes_.tolist())

if __name__ == "__main__":
    if len(sys.argv) != 3: print("Usage: python Tree.py <train.csv> <eval_test_data.csv>"); sys.exit(1)
    
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'Data', 'knight')
    train_f = os.path.join(data_dir, sys.argv[1])
    test_f = os.path.join(data_dir, sys.argv[2]) # This will be Ex05_Validation_knight.csv
    
    try: run_tree_pipeline(train_f, test_f)
    except Exception as e: print(f"An error occurred: {e}"); sys.exit(1)
