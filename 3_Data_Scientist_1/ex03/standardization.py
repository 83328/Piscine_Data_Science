import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler

def standardize_data(train_df, test_df):
    # Select numeric columns, excluding 'knight' if it's there
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    if 'knight' in numeric_cols:
        numeric_cols.remove('knight')

    # Fit on train, transform both
    scaler = StandardScaler()
    train_scaled = pd.DataFrame(scaler.fit_transform(train_df[numeric_cols]), columns=numeric_cols, index=train_df.index)
    test_scaled = pd.DataFrame(scaler.transform(test_df[numeric_cols]), columns=numeric_cols, index=test_df.index)

    # Reattach the 'knight' column if present
    if 'knight' in train_df.columns:
        train_scaled['knight'] = train_df['knight']
    if 'knight' in test_df.columns:
        test_scaled['knight'] = test_df['knight']

    return train_scaled, test_scaled

def print_data_summary(df, label, is_standardized=False):
    """Prints a summary of the dataframe, mimicking the exercise's output format."""
    print(label)
    cols = df.columns.drop('knight') if 'knight' in df.columns else df.columns

    if len(cols) > 6:
        display_cols_header = list(cols[:3]) + ['...'] + list(cols[-3:])
        display_cols_data = list(cols[:3]) + list(cols[-3:]) # for actual data indexing
    else:
        display_cols_header = list(cols)
        display_cols_data = list(cols)

    # Print header row
    print(" ".join(col.ljust(12) for col in display_cols_header))

    # Print first data row and '...'
    if not df.empty:
        first_row_values = df.iloc[0][display_cols_data]
        formatted_values = [f"{v:.2f}" if isinstance(v, (int, float)) else str(v) for v in first_row_values]

        if len(cols) > 6:
            final_display_vals = formatted_values[:3] + ['...'] + formatted_values[3:]
        else:
            final_display_vals = formatted_values

        print(" ".join(val.ljust(12) for val in final_display_vals))
        print("...")


def plot_dataset(ax, df, title, separate_clusters=True):
    """Plots a scatter plot of the data, optionally separating clusters."""
    # Use first two numeric features
    features = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'knight' in features:
        features.remove('knight')

    if len(features) < 2:
        print(f"Not enough numeric features to plot for {title}")
        return

    x_feature, y_feature = features[0], features[1]

    if 'knight' in df.columns and separate_clusters:
        sith = df[df['knight'] == 'Sith']
        jedi = df[df['knight'] == 'Jedi']
        ax.scatter(sith[x_feature], sith[y_feature], c='red', alpha=0.6, label='Sith')
        ax.scatter(jedi[x_feature], jedi[y_feature], c='blue', alpha=0.6, label='Jedi')
    else:
        # Plot all points without distinguishing by 'knight' type
        ax.scatter(df[x_feature], df[y_feature], c='green', alpha=0.6, label='Knight')

    ax.set_title(title)
    ax.set_xlabel("???") # As per example, specific feature names are omitted for these axes
    ax.set_ylabel("???") # As per example, specific feature names are omitted for these axes
    if separate_clusters or 'knight' not in df.columns: # Only show legend if relevant
        ax.legend()

def main():
    # Define paths
    current_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..')) # 
    data_dir = os.path.join(project_root, 'Data', 'knight')

    train_file_path = os.path.join(data_dir, 'Train_knight.csv')
    test_file_path = os.path.join(data_dir, 'Test_knight.csv')

    if not os.path.exists(train_file_path):
        print(f"Error: Train_knight.csv not found at {train_file_path}")
        return
    if not os.path.exists(test_file_path):
        print(f"Error: Test_knight.csv not found at {test_file_path}")
        return

    train_df = pd.read_csv(train_file_path)
    test_df = pd.read_csv(test_file_path)

    # Print original data for comparison (as per example output)
    print_data_summary(train_df, "Original Train_knight.csv")
    print_data_summary(test_df, "Original Test_knight.csv")

    # Standardize data
    train_std, test_std = standardize_data(train_df, test_df)

    # Print standardized summaries (as per example output)
    print_data_summary(train_std, "Standardized Train_knight.csv", is_standardized=True)
    print_data_summary(test_std, "Standardized Test_knight.csv", is_standardized=True)

    print("\nPlotting one graph with standardized data (Train_knight - Separated Clusters).")
    fig, ax = plt.subplots(figsize=(7, 7))
    plot_dataset(ax, train_std, "Train_knight (Standardized)", separate_clusters=True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()