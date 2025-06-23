import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler

def normalize_data(train_df, test_df):
    # Select numeric columns, excluding 'knight' if it's there
    numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
    if 'knight' in numeric_cols:
        numeric_cols.remove('knight')

    # Fit on train, transform both
    scaler = MinMaxScaler()
    train_norm = pd.DataFrame(scaler.fit_transform(train_df[numeric_cols]), columns=numeric_cols, index=train_df.index)
    test_norm = pd.DataFrame(scaler.transform(test_df[numeric_cols]), columns=numeric_cols, index=test_df.index)

    # Reattach the 'knight' column if present
    if 'knight' in train_df.columns:
        train_norm['knight'] = train_df['knight']
    if 'knight' in test_df.columns:
        test_norm['knight'] = test_df['knight']

    return train_norm, test_norm

def print_normalized_data(df, label):
    print(label)
    cols = df.columns.drop('knight') if 'knight' in df.columns else df.columns
    if len(cols) > 6:
        # Select first 3 and last 3 columns for display
        display_cols_header = list(cols[:3]) + ['...'] + list(cols[-3:])
        display_cols_data = list(cols[:3]) + list(cols[-3:]) # for actual data indexing
    else:
        display_cols_header = list(cols)
        display_cols_data = list(cols)

    print(" ".join(col.ljust(12) for col in display_cols_header))
    if not df.empty:
        first_row = df.iloc[0][display_cols_data]
        # Ensure values are formatted correctly, especially for the '...' placeholder
        display_vals = [f"{v:.2f}" if isinstance(v, (int, float)) else str(v) for v in first_row]
        
        # Reconstruct with '...' if needed for display
        if len(cols) > 6:
            final_display_vals = display_vals[:3] + ['...'] + display_vals[3:]
        else:
            final_display_vals = display_vals

        print(" ".join(val.ljust(12) for val in final_display_vals))
        print("...")

def plot_dataset(ax, df, title, separate_clusters=True):
    # Use first two numeric features. You might want to choose these based on correlation for better visualization.
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
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
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

    # Normalize data
    train_norm, test_norm = normalize_data(train_df, test_df)

    # Print normalized summaries
    print_normalized_data(train_norm, "Train (Normalized)")
    print_normalized_data(test_norm, "Test (Normalized)")

    # This corresponds to "Train Visually Separated (Normalized)".
    fig, ax = plt.subplots(1, 1, figsize=(8, 6)) # Create a single subplot

    # Plot 1: Train Visually Separated (Normalized)
    plot_dataset(ax, train_norm, "Train_knight (Normalized)", separate_clusters=True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()