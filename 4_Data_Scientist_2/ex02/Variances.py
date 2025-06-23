import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def calculate_and_plot_variances(data_filepath):
    """
    Loads data, calculates individual and cumulative explained variances using PCA,
    prints them, and displays a cumulative variance plot.

    Args:
        data_filepath (str): The full path to the CSV data file.
    """
    try:
        # Load the data from the specified CSV file
        df = pd.read_csv(data_filepath)
    except FileNotFoundError:
        print(f"Error: Data file not found at '{data_filepath}'")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{data_filepath}' is empty.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV file '{data_filepath}': {e}")
        sys.exit(1)

    # Separate features from the target variable if 'knight' column exists
    # Assuming 'knight' is the target and other columns are features.
    # If 'knight' is not present or if all columns are features, adjust accordingly.
    if 'knight' in df.columns:
        X = df.drop('knight', axis=1)
    else:
        # If 'knight' column is not found, assume all columns are features
        # or handle based on knowledge of your dataset structure.
        print("Warning: 'knight' column not found. Assuming all columns are features for variance calculation.")
        X = df.copy()

    # Ensure only numeric columns are used for PCA
    numeric_X = X.select_dtypes(include=np.number)

    if numeric_X.empty:
        print("Error: No numeric features found in the dataset after dropping 'knight' (if present).")
        sys.exit(1)

    # Standardize the data before applying PCA
    # PCA is affected by scale, so it's good practice to standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(numeric_X)

    # Initialize PCA with all components to get individual explained variances
    pca = PCA(n_components=None) # n_components=None means keep all components
    pca.fit(X_scaled)

    # Calculate explained variance ratio for each principal component, as percentage
    explained_variances_ratio = pca.explained_variance_ratio_ * 100

    # Calculate cumulative explained variance ratio, as percentage
    cumulative_explained_variances = np.cumsum(explained_variances_ratio)

    # --- Print Variances (Percentage) ---
    print("Variances (Percentage):")
    print("[", end="")
    for i, var in enumerate(explained_variances_ratio):
        print(f"{var:.11e}", end="")
        if i < len(explained_variances_ratio) - 1:
            print(" ", end="")
    print("]\n")

    # --- Print Cumulative Variances (Percentage) ---
    print("Cumulative Variances (Percentage):")
    print("[", end="")
    for i, cum_var in enumerate(cumulative_explained_variances):
        print(f"{cum_var:.8f}", end="") # Use .8f for consistency with subject's example output precision
        if i < len(cumulative_explained_variances) - 1:
            print(" ", end="")
    print("]\n")

    # --- Display Graph ---
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(cumulative_explained_variances) + 1), cumulative_explained_variances, marker='', linestyle='-')
    plt.xlabel('Number of components', fontsize=12)
    plt.ylabel('Explained variance (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(np.arange(0, len(cumulative_explained_variances) + 1, 5)) # Set x-ticks similar to example
    plt.yticks(np.arange(40, 105, 10)) # Set y-ticks from 40 to 105 with steps of 10
    plt.ylim(40, 105) # Ensure y-axis goes from 40 to 105
    plt.tight_layout()
    plt.show()

def main():
    if len(sys.argv) != 2:
        print("Usage: python variances.py <data_filename>")
        print("Example: python variances.py Training_knight.csv")
        sys.exit(1)

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'Data', 'knight')

    data_filename = sys.argv[1]
    data_filepath = os.path.join(data_dir, data_filename)

    print(f"Calculating and plotting variances for: {data_filepath}\n")
    calculate_and_plot_variances(data_filepath)

if __name__ == "__main__":
    main()
