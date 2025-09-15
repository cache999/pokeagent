import os
import glob
import pandas as pd
import matplotlib.pyplot as plt


def visualize_csvs_from_worlds(name):
    """Load CSV files from worlds/log/csv directory and display them as 2D image arrays."""
    # Find CSV files
    csv_files = glob.glob("worlds/log/csv/" + name + ".csv", recursive=True)
    if not csv_files:
        csv_files = glob.glob("**/*csv", recursive=True)  # Fallback search

    if not csv_files:
        print("No CSV files found!")
        return

    # Display each CSV as a 2D image
    for csv_file in csv_files:
        try:
            # Load CSV data
            data = pd.read_csv(csv_file, header=None).values

            # Create visualization
            plt.figure(figsize=(20, 12))
            plt.imshow(data, cmap='viridis', aspect='auto')
            plt.colorbar()

            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    text = plt.text(j, i, f'{data[i, j]:.2f}',
                                  ha="center", va="center", color="white", fontsize=8)


            plt.title(f'2D Array: {os.path.basename(csv_file)}')
            plt.show()

        except Exception as e:
            print(f"Error processing {csv_file}: {e}")


# Run the function
visualize_csvs_from_worlds("map_0_20250914_152725")
