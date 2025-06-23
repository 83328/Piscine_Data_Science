# IMPORTANT: Before running this script, ensure you have the 'statsmodels' library installed.
# You can install it using pip:
# pip install statsmodels

import sys
import os
import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant # Required for VIF calculation

def calculate_vif(df_features):
    """
    Calculates the Variance Inflation Factor (VIF) for each feature in a DataFrame.
    It's recommended to add a constant to the DataFrame for accurate VIF calculation
    when interpreting coefficients in a regression model.
    """
    # Add a constant column for VIF calculation if not already present.
    # VIF measures how much the variance of an estimated regression coefficient
    # is inflated due to multicollinearity. It's typically used in regression contexts
    # where an intercept (constant) is assumed.
    df_with_const = add_constant(df_features)
    
    vif_data = pd.DataFrame()
    vif_data["feature"] = df_with_const.columns
    # Skip the constant term itself when calculating VIF for features
    vif_data["VIF"] = [variance_inflation_factor(df_with_const.values, i)
                       for i in range(df_with_const.shape[1])]
    # Remove the 'const' row as it's not a feature VIF
    vif_data = vif_data[vif_data['feature'] != 'const']
    return vif_data.sort_values(by="VIF", ascending=False).reset_index(drop=True)

def select_features_by_vif(data_filepath, vif_threshold=5.0):
    """
    Performs feature selection based on VIF, iteratively removing features
    with VIF above the threshold until all remaining features are below it.

    Args:
        data_filepath (str): The full path to the CSV data file.
        vif_threshold (float): The maximum allowed VIF for features.
    """
    try:
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

    # Assuming 'knight' is the target variable and should not be included in VIF calculation
    if 'knight' in df.columns:
        features_df = df.drop('knight', axis=1)
    else:
        print("Warning: 'knight' column not found. Assuming all columns are features for VIF calculation.")
        features_df = df.copy()

    # Ensure all features are numeric for VIF calculation
    numeric_features_df = features_df.select_dtypes(include=np.number)

    if numeric_features_df.empty:
        print("Error: No numeric features found in the dataset for VIF calculation.")
        sys.exit(1)

    # Make a copy to avoid modifying the original DataFrame directly within the loop
    current_features = numeric_features_df.copy()
    
    print(f"--- Initial VIF Calculation (Threshold: {vif_threshold}) ---\n")
    vif_results = calculate_vif(current_features)
    print(vif_results.to_string(index=False)) # Use to_string to print all rows without truncation

    print("\n--- Iterative Feature Selection ---")
    iteration = 1
    while True:
        vif_results = calculate_vif(current_features)
        max_vif = vif_results["VIF"].max()

        if max_vif < vif_threshold:
            print(f"\nIteration {iteration}: All remaining features have VIF < {vif_threshold}.")
            break
        
        feature_to_remove = vif_results.iloc[0]["feature"] # Feature with the highest VIF
        current_features = current_features.drop(columns=[feature_to_remove])
        
        print(f"\nIteration {iteration}: Removing '{feature_to_remove}' (VIF: {max_vif:.2f})")
        print("Current VIFs:")
        print(calculate_vif(current_features).to_string(index=False)) # Print VIFs after removal
        iteration += 1

        if current_features.empty:
            print("\nNo features left after VIF selection.")
            break

    print("\n--- Final Selected Features (VIF < 5) ---")
    if not current_features.empty:
        print("Remaining features:")
        for feature in current_features.columns:
            print(f"- {feature}")
    else:
        print("No features remained after VIF filtering.")

def main():
    if len(sys.argv) != 2:
        print("Usage: python Feature_Selection.py <data_filename>")
        print("Example: python Feature_Selection.py Training_knight.csv")
        sys.exit(1)

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'Data', 'knight')

    data_filename = sys.argv[1]
    data_filepath = os.path.join(data_dir, data_filename)

    select_features_by_vif(data_filepath)

if __name__ == "__main__":
    main()
