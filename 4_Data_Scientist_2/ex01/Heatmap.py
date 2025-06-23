import sys
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder # Import LabelEncoder

def generate_correlation_heatmap(data_filepath, title="Correlation Heatmap"):
    """
    Loads data from a CSV file, calculates the correlation matrix,
    and displays it as a heatmap. Includes the 'knight' column after encoding.

    Args:
        data_filepath (str): The full path to the CSV data file.
        title (str): The title for the heatmap plot.
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

    # --- Start of changes to include 'knight' ---
    if 'knight' in df.columns:
        # Initialize LabelEncoder
        le = LabelEncoder()
        # Fit and transform the 'knight' column to numerical values
        # e.g., 'Jedi' -> 0, 'Sith' -> 1 (or vice versa)
        df['knight_encoded'] = le.fit_transform(df['knight'])
        
        # Now, select all numeric columns INCLUDING the newly encoded 'knight' column.
        # It's safer to drop the original 'knight' column if it's still present before selecting numerics
        # to avoid any issues if pandas somehow infers it as numeric (though unlikely for strings).
        df_for_corr = df.drop(columns=['knight']) if 'knight' in df.columns else df
        correlation_matrix = df_for_corr.select_dtypes(include=np.number).corr()
    else:
        print("Warning: 'knight' column not found in the data. Generating heatmap for numeric features only.")
        # If 'knight' is not present, proceed as before, only with existing numeric columns.
        correlation_matrix = df.select_dtypes(include=np.number).corr()
    # --- End of changes to include 'knight' ---

    # --- RENAME FOR PLOTTING ---
    # After calculating the correlation, rename 'knight_encoded' to 'knight' for a cleaner plot label.
    if 'knight_encoded' in correlation_matrix.columns:
        correlation_matrix = correlation_matrix.rename(
            columns={'knight_encoded': 'knight'},
            index={'knight_encoded': 'knight'}
        )
    # --- END RENAME ---

    # Set up the matplotlib figure and axes for the heatmap
    plt.figure(figsize=(12, 10)) # Adjust size for better readability with more features

    # Create the heatmap using seaborn
    sns.heatmap(
        correlation_matrix,
        cmap='rocket', # Colormap chosen to resemble the subject's example)
        annot=False,     # No annotations (numbers) on the heatmap cells, as per subject image
        fmt=".2f",       # Format annotations to 2 decimal places if annot=True
        linewidths=.5,   # Add lines between cells for clarity
        cbar_kws={"shrink": .75}, # Shrink colorbar to fit better
        vmin=-1,         # Ensure color scale covers full correlation range (-1 to 1)
        vmax=1
    )

    # Set the title of the plot
    plt.title(title, fontsize=16)

    # Adjust layout to prevent labels from being cut off
    plt.tight_layout()

    # Display the plot
    plt.show()

def main():
    if len(sys.argv) != 2:
        print("Usage: python Heatmap.py <data_filename>")
        print("Example: python Heatmap.py Training_knight.csv")
        sys.exit(1)

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'Data', 'knight')

    data_filename = sys.argv[1]
    data_filepath = os.path.join(data_dir, data_filename)

    print(f"Generating correlation heatmap...")
    generate_correlation_heatmap(data_filepath, title="Correlation Heatmap")

if __name__ == "__main__":
    main()
