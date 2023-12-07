import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from statistics import mean, stdev
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from bokeh.plotting import figure, save

gender_csv = pd.read_excel("datasets/Original_Datasets/Individual sex_Microclosure counts_time closed.xlsx", header = 1)
gender_dict = dict(zip(gender_csv.ID , gender_csv["Sex (histology)"] ))
gender_dict

csv_path = "datasets/outputs/train_df_nt_after_norm.csv"
train_df = pd.read_csv(csv_path, index_col=0)


for unique_id in tqdm(train_df['ID'].unique()):
  id_rows = train_df[train_df['ID'] == unique_id]

  # Plot and save each sequence
  plt.figure(figsize=(35, 5))
  sns.lineplot(data=id_rows['sequence'])
  plt.savefig(f'plots3/{unique_id}_{gender_dict[unique_id]}.png')
  plt.close()





    