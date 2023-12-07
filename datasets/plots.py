import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.signal import savgol_filter
import matplotlib as mpl

mpl.rcParams['agg.path.chunksize'] = 10000

print('Importing df')

# Read the CSV file
df = pd.read_csv('/gpfs/scratch/rayen/Oysters/datasets/train_df2_after_norm_mem_reduce.csv')

grouped = df.groupby('ID')

# Define the initial directory name
dir_name = 'plots'
dir_counter = 0

# Check for the existence of the initial directory and increment the counter
while os.path.exists(dir_name):
    dir_counter += 1
    dir_name = f'plots{dir_counter}'

# Create the new directory
os.makedirs(dir_name)

print('Plotting')
# Plot and save each sequence within the groups
for name, group in grouped:
    print(f'name: {name}, group: {group}')

    plt.figure(figsize=(100, 30))
    plt.plot(group.index, group['Sequence'])  # Plot the smoothed sequence
    plt.xlabel('Index', fontsize=35)  # Adjust the fontsize for x-axis label
    plt.ylabel('Sequence', fontsize=35)  # Adjust the fontsize for y-axis label
    plt.title(f'Sequence for {name}', fontsize=35)  # Adjust the fontsize for title

    # Define the file name within the new directory
    file_name = f'{dir_name}/{name}_{group["label"].iloc[0]}.png'

    plt.savefig(file_name)
    plt.close()

print(f"Plots saved in the '{dir_name}' folder.")


"""# Separate the data into label 1 and label 0
label_1_data = df[df['label'] == 1]
label_0_data = df[df['label'] == 0]

# Plot and save both label 1 and label 0 sequences on the same plot, starting from x = 0
plt.figure(figsize=(8, 6))
plt.plot(range(len(label_1_data)), label_1_data['Sequence'], label='Label 1', color='blue')
plt.plot(range(len(label_0_data)), label_0_data['Sequence'], label='Label 0', color='red')
plt.xlabel('Index')
plt.ylabel('Sequence')
plt.legend()
plt.title('Sequences with Labels 1 and 0')

# Save the plot
file_name = 'plots/all_sequences0.png'
plt.savefig(file_name)
plt.close()


print(f"Plot saved as '{file_name}' in the 'plots' folder.")
"""