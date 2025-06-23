import sys, os, pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score
import warnings; warnings.filterwarnings("ignore", category=FutureWarning)

def run_knn_pipeline(main_train_arg, main_test_arg):
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'Data', 'knight')
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "KNN.txt")

    # Load data
    try:
        train_df = pd.read_csv(os.path.join(data_dir, main_train_arg))
        test_df = pd.read_csv(os.path.join(data_dir, main_test_arg))
        ex05_train_df = pd.read_csv(os.path.join(data_dir, "Ex05_Training_knight.csv"))
        ex05_val_df = pd.read_csv(os.path.join(data_dir, "Ex05_Validation_knight.csv"))
    except Exception as e: print(f"Error loading data: {e}"); sys.exit(1)

    # Split features/targets
    X_train, y_train = train_df.drop(columns=['knight']), train_df['knight']
    X_ex05_train, y_ex05_train = ex05_train_df.drop(columns=['knight']), ex05_train_df['knight']
    X_ex05_val, y_ex05_val = ex05_val_df.drop(columns=['knight']), ex05_val_df['knight']
    X_test = test_df.copy()

    # Preprocessing (scale numeric, one-hot encode categorical)
    combined_X = pd.concat([X_train, X_ex05_train, X_ex05_val, X_test], ignore_index=True)
    numeric_cols = combined_X.select_dtypes(include=np.number).columns.tolist()
    scaler = StandardScaler(); combined_X[numeric_cols] = scaler.fit_transform(combined_X[numeric_cols])
    combined_X_processed = pd.get_dummies(combined_X)

    # Split processed data back
    idx=0; X_train_p = combined_X_processed.iloc[idx:len(X_train)]; idx+=len(X_train)
    X_ex05_train_p = combined_X_processed.iloc[idx:idx+len(X_ex05_train)]; idx+=len(X_ex05_train)
    X_ex05_val_p = combined_X_processed.iloc[idx:idx+len(X_ex05_val)]; idx+=len(X_ex05_val)
    X_test_p = combined_X_processed.iloc[idx:idx+len(X_test)]

    # Label encode targets
    le = LabelEncoder(); le.fit(pd.concat([y_train, y_ex05_train, y_ex05_val]))
    y_train_e, y_ex05_train_e, y_ex05_val_e = le.transform(y_train), le.transform(y_ex05_train), le.transform(y_ex05_val)

    # K-tuning on EX05_Training/Validation
    k_vals = range(1, min(51, len(X_ex05_train_p)+1)); accuracies = []; f1_scores_tuning = []; best_k = 1; best_f1_validation = -1.0
    for k in k_vals:
        knn = KNeighborsClassifier(n_neighbors=k); knn.fit(X_ex05_train_p, y_ex05_train_e)
        y_pred_val = knn.predict(X_ex05_val_p); f1 = f1_score(y_ex05_val_e, y_pred_val, average='weighted', zero_division=0)
        accuracies.append(accuracy_score(y_ex05_val_e, y_pred_val)); f1_scores_tuning.append(f1) # Store F1
        if f1 > best_f1_validation: best_f1_validation, best_k = f1, k

    # Plot K-tuning results
    plt.figure(figsize=(10,6)); plt.plot(k_vals, accuracies, marker='o'); plt.title('KNN Accuracy vs. K'); plt.xlabel('K'); plt.ylabel('Accuracy'); plt.grid(True); plt.show()

    # Final model training & prediction
    final_knn = KNeighborsClassifier(n_neighbors=best_k); final_knn.fit(X_train_p, y_train_e)
    y_pred_final_e = final_knn.predict(X_test_p); y_preds_final = le.inverse_transform(y_pred_final_e)

    # Save predictions
    with open(out_path, 'w') as f: f.write('\n'.join(y_preds_final))
    print(f"Predictions saved to {out_path}")

    # Report performance (Best F1 and Mean F1 from validation tuning)
    print(f"----------------------------------------------------------\n")
    print(f"Best F1-Score from Validation Set (K={best_k}): {best_f1_validation:.4f}")
    print(f"Mean F1-Score across K values (Validation Set): {np.mean(f1_scores_tuning):.4f}\n")
    print(f"----------------------------------------------------------\n")

if __name__ == "__main__":
    if len(sys.argv) != 3: print("Usage: python KNN.py <main_train_data.csv> <main_test_data.csv>"); sys.exit(1)
    try: run_knn_pipeline(sys.argv[1], sys.argv[2])
    except Exception as e: print(f"An error occurred: {e}"); sys.exit(1)
