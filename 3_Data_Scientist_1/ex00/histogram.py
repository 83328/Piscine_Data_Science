import pandas as pd
import matplotlib.pyplot as plt
import os

# Define paths to CSV files
data_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'Data', 'knight')
train_path = os.path.join(data_dir, 'Train_knight.csv')
test_path = os.path.join(data_dir, 'Test_knight.csv')

try:
    # Load datasets
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Basic exploratory analysis (summary statistics)
    print("Train Dataset Summary:")
    print(train_df.describe())
    print("\nTest Dataset Summary:")
    print(test_df.describe())

    # Get common numerical columns (exclude 'knight' from Train)
    common_columns = [col for col in test_df.columns if col in train_df.columns and col != 'knight']
    if not common_columns:
        raise Exception("No valid common columns found for plotting. Check column names.")

    # First Popup: Histograms for Test_knight.csv (6x5 grid)
    fig1, axes1 = plt.subplots(6, 5, figsize=(10, 8))
    axes1 = axes1.flatten()

    for idx, col in enumerate(common_columns):
        ax = axes1[idx]
        ax.clear()
        test_df[col].hist(bins=30, color='green', alpha=0.7, ax=ax, label='Knight')
        ax.set_title(col, fontsize=9)
        ax.legend(fontsize=6)
        ax.tick_params(axis='both', labelsize=7)

    # Hide unused axes
    for j in range(idx + 1, len(axes1)):
        fig1.delaxes(axes1[j])

    plt.subplots_adjust(hspace=1.5, wspace=0.3)  # Increased height spacing between rows
    plt.tight_layout(pad=1.0)  # Add padding around the entire figure
    plt.show()


    # Second Popup: Overlapping Histograms for Test_knight.csv and Train_knight.csv (6x5 grid)
    fig2, axes2 = plt.subplots(6, 5, figsize=(10, 8))
    axes2 = axes2.flatten()

    for idx, col in enumerate(common_columns):
        ax = axes2[idx]
        ax.clear()
        ax.hist(train_df[col], bins=30, color='red', alpha=0.5, label='Sith')
        ax.hist(test_df[col], bins=30, color='blue', alpha=0.5, label='Jedi')
        ax.set_title(col, fontsize=9)
        ax.legend(fontsize=6)
        ax.tick_params(axis='both', labelsize=7)

    # Hide unused axes
    for j in range(idx + 1, len(axes2)):
        fig2.delaxes(axes2[j])

    plt.subplots_adjust(hspace=1.5, wspace=0.3)  # Increased height spacing between rows
    plt.tight_layout(pad=1.0)  # Add padding around the entire figure
    plt.show()


except FileNotFoundError as e:
    print(f"Error: File not found - {e}")
except Exception as e:
    print(f"Error: {e}")
    exit(1)
