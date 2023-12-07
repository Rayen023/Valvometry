import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
from tqdm import tqdm
mpl.rcParams['agg.path.chunksize'] = 10000

dir_name, dir_counter = 'plots', 0
while os.path.exists(dir_name): 
    dir_counter += 1
    dir_name = f'plots{dir_counter}'
os.makedirs(dir_name)

# Read the CSV file
df = pd.read_csv('/gpfs/scratch/rayen/Oysters/datasets/train_df2_after_norm_mem_reduce.csv')
grouped = df.groupby('ID')


print('Plotting')
# Plot and save each sequence within the groups
for name, group in grouped:
    week = 0
    for i in range(0, len(group), int(len(group) * 7 / 35)):  # 35 jours, 860 hours #30966466 ms

        print(f'name : {name}, group : {group}')

        plt.figure(figsize=(100, 30))
        # Convert the slice indices to integers using int()
        plt.plot(group.index[i:i + int(len(group) * 7 / 35)], group['Sequence'][i:i + int(len(group) * 7 / 35)])
        plt.xlabel('Index', fontsize=35)  # Adjust the fontsize for x-axis label
        plt.ylabel('Sequence', fontsize=35)  # Adjust the fontsize for y-axis label
        plt.title(f'Sequence for {name}', fontsize=35)  # Adjust the fontsize for title

        # Define the file name within the new directory
        file_name = f'{dir_name}/{name}_g{group["label"].iloc[0]}_w{week}.png'

        plt.savefig(file_name)
        plt.close()
        week += 1


print(f"Plots saved in the '{dir_name}' folder.")
