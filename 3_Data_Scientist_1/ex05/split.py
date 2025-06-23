import pandas as pd
import os
from sklearn.model_selection import train_test_split

def split_data(input_file_path, train_output_path, val_output_path, test_size_ratio=0.2, random_seed=42):
    """
    Splits an input CSV file into training and validation sets.

    Args:
        input_file_path (str): Path to the input CSV file (e.g., 'Train_knight.csv').
        train_output_path (str): Path to save the training CSV file.
        val_output_path (str): Path to save the validation CSV file.
        test_size_ratio (float): The proportion of the dataset to include in the validation split.
        random_seed (int): Controls the shuffling applied to the data before applying the split.
                           Pass an int for reproducible output across multiple function calls.
    """
    if not os.path.exists(input_file_path):
        print(f"Error: Input file not found at {input_file_path}")
        return

    # Load the data
    df = pd.read_csv(input_file_path)

    # Perform the split
    # 'train_test_split' returns two DataFrames: training and (test/validation)
    train_df, val_df = train_test_split(df, test_size=test_size_ratio, random_state=random_seed)

    # Save the split data to new CSV files
    train_df.to_csv(train_output_path, index=False)
    val_df.to_csv(val_output_path, index=False)

    print(f"Successfully split '{os.path.basename(input_file_path)}'.")
    print(f"Training data saved to '{os.path.basename(train_output_path)}' ({len(train_df)} rows).")
    print(f"Validation data saved to '{os.path.basename(val_output_path)}' ({len(val_df)} rows).")
    print(f"Split ratio: {100 * (1 - test_size_ratio):.0f}% Training / {100 * test_size_ratio:.0f}% Validation.")


def main():
    # Define paths
    current_dir = os.path.dirname(__file__)
    # Assuming 'split.py' is in 'ex05/'
    # And 'Train_knight.csv' is in 'Data/knight' relative to the project root
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    data_dir = os.path.join(project_root, 'Data', 'knight')

    input_file = os.path.join(data_dir, 'Train_knight.csv')
    train_output_file = os.path.join(data_dir, 'Ex05_Training_knight.csv')
    val_output_file = os.path.join(data_dir, 'Ex05_Validation_knight.csv')

    # Example: 75% for training, 25% for validation
    split_percentage = 0.25

    split_data(input_file, train_output_file, val_output_file, test_size_ratio=split_percentage)

if __name__ == "__main__":
    main()
