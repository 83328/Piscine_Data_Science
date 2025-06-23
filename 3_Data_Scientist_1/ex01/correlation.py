import pandas as pd
import os
import numpy as np # Import numpy for pd.isna

def calculate_and_print_correlations(train_path):
    try:
        # Load dataset
        train_df = pd.read_csv(train_path)

        # Check and convert 'knight' column to numeric if categorical
        if train_df['knight'].dtype == 'object':
            # Map Jedi to 1 and Sith to 0. The specific values (0/1)
            # don't affect the absolute correlation, but determine the sign.
            knight_mapping = {'Jedi': 1, 'Sith': 0}
            train_df['knight'] = train_df['knight'].map(knight_mapping)
        
        # Compute correlation matrix, focusing on numeric columns
        # numeric_only=True is a good practice to avoid warnings/errors if non-numeric non-target columns exist
        correlation_matrix = train_df.corr(numeric_only=True)

        # Extract correlations with 'knight' and drop self-correlation
        if 'knight' in correlation_matrix.columns:
            knight_correlations = correlation_matrix['knight'].drop('knight')
        else:
            print("Error: 'knight' column not found or not numeric after mapping. Cannot compute correlations.")
            return

        # Consider all features that will be printed in the output, including 'knight' itself.
        all_feature_names_for_max_len = ['knight'] + list(knight_correlations.index)
        max_len = max(len(feat) for feat in all_feature_names_for_max_len)

        # Print the knight self-correlation first
        print(f"knight".ljust(max_len) + "  1.000000")

        # Sort the correlations by their absolute value in descending order
        sorted_correlations = knight_correlations.abs().sort_values(ascending=False)

        # Then display correlations for the features, sorted by absolute value
        for feat in sorted_correlations.index:
            # Get the original correlation value (not the absolute one)
            original_corr_value = knight_correlations[feat]
            
            # Ensure the correlation value is not NaN before printing
            if not pd.isna(original_corr_value):
                print(f"{feat.ljust(max_len)}  {original_corr_value:.6f}")

    except FileNotFoundError as e:
        print(f"Error: The file was not found - {e}")
    except KeyError as e:
        print(f"Error: A required column was not found in the dataset - {e}.")
    except ValueError as e:
        print(f"Error: Data conversion issue, possibly non-numeric values in numeric columns: {e}.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def main():
    current_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    data_dir = os.path.join(project_root, 'Data', 'knight')
    train_path = os.path.join(data_dir, 'Train_knight.csv')

    calculate_and_print_correlations(train_path)

if __name__ == "__main__":
    main()