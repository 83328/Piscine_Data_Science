import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

def plot_dataset(ax, df, title, x_feature, y_feature, separate_clusters=True):
    """
    Plots a scatter plot of the data on the given axes.
    This function handles both 'clustered' (Sith/Jedi) and 'mixed' (all Knight) plots.

    :param ax: Matplotlib axes object to plot on.
    :param df: DataFrame containing the data.
    :param title: Title for the plot.
    :param x_feature: Name of the column to use for the x-axis.
    :param y_feature: Name of the column to use for the y-axis.
    :param separate_clusters: If True, attempts to separate 'Sith' and 'Jedi' by color.
                              If False, plots all points as 'Knight' in a single color.
    """
    # Ensure the chosen features exist in the DataFrame
    if x_feature not in df.columns or y_feature not in df.columns:
        print(f"Error: Features '{x_feature}' or '{y_feature}' not found in DataFrame for plot '{title}'.")
        # Attempt to find alternative numeric features if specific ones are missing
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if 'knight' in numeric_cols:
            numeric_cols.remove('knight')
        if len(numeric_cols) >= 2:
            x_feature = numeric_cols[0]
            y_feature = numeric_cols[1]
            print(f"Attempting to plot with '{x_feature}' and '{y_feature}' instead due to missing features.")
        else:
            print("Not enough numeric features to plot. Skipping this plot.")
            return

    # Plotting logic based on 'separate_clusters' and 'knight' column presence
    if 'knight' in df.columns and separate_clusters:
        # Separate plotting for 'Sith' and 'Jedi'
        sith_data = df[df['knight'] == 'Sith']
        jedi_data = df[df['knight'] == 'Jedi']
        ax.scatter(sith_data[x_feature], sith_data[y_feature], c='red', alpha=0.6, label='Sith')
        ax.scatter(jedi_data[x_feature], jedi_data[y_feature], c='blue', alpha=0.6, label='Jedi')
    else:
        # Plot all points as a single 'Knight' cluster
        ax.scatter(df[x_feature], df[y_feature], color='green', alpha=0.6, label='Knight')

    # Set plot title and axis labels
    ax.set_title(title)
    ax.set_ylabel("???") # As per exercise example, axis labels are generic
    ax.legend() # Display legend for clarity

def main():
    # Define paths to CSV files for training and testing data
    current_dir = os.path.dirname(__file__)
    # Navigate up two directories from 'ex02' to the project root, then down to 'Data/knight'
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
    data_dir = os.path.join(project_root, 'Data', 'knight')

    train_path = os.path.join(data_dir, 'Train_knight.csv')
    test_path = os.path.join(data_dir, 'Test_knight.csv')

    # Verify that the data files exist before attempting to load them
    if not os.path.exists(train_path):
        print(f"Error: Train_knight.csv not found at {train_path}. Please ensure the file path is correct.")
        return
    if not os.path.exists(test_path):
        print(f"Error: Test_knight.csv not found at {test_path}. Please ensure the file path is correct.")
        return

    # Load datasets into pandas DataFrames
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # --- Determine the features to use for plotting ---
    numeric_features = train_df.select_dtypes(include=np.number).columns.tolist()
    if 'knight' in numeric_features:
        numeric_features.remove('knight') # 'knight' is the target, not a feature for plotting axes


    # Choose two distinct pairs of features for plotting
    x1_feature, y1_feature = numeric_features[4], numeric_features[5]
    x2_feature, y2_feature = numeric_features[1], numeric_features[2] # Use next two for the second pair

    print(f"Plotting left column graphs using '{x1_feature}' for x-axis and '{y1_feature}' for y-axis.")
    print(f"Plotting right column graphs using '{x2_feature}' for x-axis and '{y2_feature}' for y-axis.")


    # Create a 2x2 subplot grid to display the four required graphs
    fig, axes = plt.subplots(2, 2, figsize=(14, 12)) # Adjust figsize for better readability if needed
    axes = axes.flatten() # Flatten the 2x2 array of axes for easier iteration/indexing

    # --- TOP ROW: Both visually separated (clustered) ---

    # Plot 1 (Top-Left): Train set - Visually Separated (Clustered) with Feature Pair 1
    plot_dataset(axes[0], train_df, f"Train (Clustered) - Pair 1", x1_feature, y1_feature, separate_clusters=True)

    # Plot 2 (Top-Right): Train set - Visually Separated (Clustered) with Feature Pair 2
    plot_dataset(axes[1], train_df, f"Train (Clustered) - Pair 2", x2_feature, y2_feature, separate_clusters=True)

    # --- BOTTOM ROW: Both mixed ---

    # Plot 3 (Bottom-Left): Test set - Mixed with Feature Pair 1
    plot_dataset(axes[2], test_df, f"Test (Mixed) - Pair 1", x1_feature, y1_feature, separate_clusters=False)

    # Plot 4 (Bottom-Right): Test set - Mixed with Feature Pair 2
    plot_dataset(axes[3], test_df, f"Test (Mixed) - Pair 2", x2_feature, y2_feature, separate_clusters=False)

    # Adjust layout to prevent titles/labels from overlapping
    plt.tight_layout()

    # Display the plot window
    plt.show()

    # Optional: Save the plot as an image file (uncomment if desired)
    # plt.savefig("points_plot.png")

if __name__ == "__main__":
    main()
