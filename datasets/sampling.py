import pandas as pd
import numpy as np
import tqdm
import glob
from sklearn import preprocessing


# Get genders dict
gender_csv = pd.read_excel("Original_Datasets/Individual sex_Microclosure counts_time closed.xlsx", header = 1)
gender_dict = dict(zip(gender_csv.ID , gender_csv["Sex (histology)"] ))
gender_dict

module = ['A', 'C', 'D', 'E', 'F', 'G','H','I']

temp_dfs = []

for letter in tqdm.tqdm(module):
    raw_data_files = sorted(glob.glob(f"Original_Datasets/Raw valvo data files_2019-2020_Second experiment (Jeff)/{letter}*.csv"))
    df = pd.concat([pd.read_csv(raw_data_files[0], header=16, index_col=0), pd.read_csv(raw_data_files[1], header=16, index_col=0)]).reset_index(drop=True)
    length = df.shape[0]
    #length = length / 100
    length = int(length - (length % 60))
    print(length)
    temp = pd.DataFrame()
    temp["sequence"] = [np.nan for _ in range(length * 4)]
    temp["ID"] = [np.nan for _ in range(length * 4)]
    temp["label"] = [np.nan for _ in range(length * 4)]
    for i in range(4):
        ID = f"{letter}{i+1}"
        if ID in gender_dict:
            temp.loc[i*length : (i+1)*length - 1, "sequence"] = df[f"CH_{i+1}"][:length].values
            temp.loc[i*length : (i+1)*length - 1, "ID"] = ID
            temp.loc[i*length : (i+1)*length - 1, "label"] = gender_dict[ID]
    temp.dropna(inplace = True)
    temp_dfs.append(temp)
merged_df = pd.concat(temp_dfs).reset_index(drop = True)


# encode label :
merged_df["label"] =  merged_df["label"].map({"F":0,"M":1})

from datetime import timedelta

# Function to add time column
def add_time_column(group):
    group['Time'] = pd.to_datetime('00:00:00') + group.groupby('ID').cumcount() * timedelta(seconds=0.1)
    return group

"""
Yearly frequency: 'Y'
Quarterly frequency: 'Q'
Monthly frequency: 'M'
Weekly frequency: 'W'
Daily frequency: 'D'
Hourly frequency: 'H'
Minutes frequency: 'Min'
Seconds frequency: 'S'
Milliseconds frequency: 'L'
Microseconds frequency: 'U'
Nanoseconds frequency: 'N'

"""

merged_df = merged_df.groupby('ID', group_keys = False).apply(add_time_column).reset_index(drop=True)

# Define the downsampling interval
interval = '1S'

# Convert Time to pandas DatetimeIndex
merged_df['Time'] = pd.to_datetime(merged_df['Time'])

# Group by ID and Time using Grouper with the specified interval and compute the mean of each group
#df_resampled = merged_df.groupby(['ID', pd.Grouper(key='Time', freq=interval)] , group_keys = False).agg({'sequence': ['mean' ,'std'], 'label':'first'}).reset_index()

df_resampled = merged_df

def normalize_sq(group):
    X = group['sequence']
    normalizer = preprocessing.MinMaxScaler(feature_range = (0,1))
    X_normalized = normalizer.fit_transform(X.values.reshape(-1, 1))
    group['sequence'] = X_normalized
    return group

df_resampled = df_resampled.groupby('ID', group_keys = False).apply(normalize_sq).reset_index(drop=True)

#df_resampled.sequence.plot(kind = "kde")

df_resampled.drop(['Time'], axis = 1, inplace = True)

df_resampled.to_csv("outputs/train_df_nt_after_norm.csv")

print(df_resampled)
