import pandas as pd
import numpy as np
import glob
import tqdm
from sklearn import preprocessing

raw_data_files = sorted(glob.glob(f"Datasets2/*.csv"))
raw_data_files

# Get genders dict
gender_csv = pd.read_excel(
    "Datasets2/ERB Oyster Conditioning - Results Winter 2019.xlsx", header=0
)
gender_dict = dict(zip(gender_csv.ID, gender_csv["Sex"]))
gender_dict

modules = ["A", "B", "C", "D", "E"]

output_dict = {}

index = 1
for module in modules:
    for i in range(1, 5):
        key = module + str(i)
        if index in gender_dict:
            value = gender_dict[index]
            if "female" in value:
                output_dict[key] = "F"
            elif "male" in value:
                output_dict[key] = "M"

        index += 1

print(output_dict)

temp_dfs = []

for letter in tqdm.tqdm(modules):
    raw_data_file = sorted(glob.glob(f"Datasets2/*{letter}.csv"))
    print(raw_data_file)
    df = pd.read_csv(raw_data_file[0], header=16, index_col=0)
    length = df.shape[0]
    # length = length / 100
    # length = int(length - (length % 3000))
    # print(length)
    temp = pd.DataFrame()
    temp["sequence"] = [np.nan for _ in range(length * 4)]
    temp["ID"] = [np.nan for _ in range(length * 4)]
    temp["label"] = [np.nan for _ in range(length * 4)]
    for i in range(4):
        ID = f"{letter}{i+1}"
        if ID in output_dict:
            temp.loc[i * length : (i + 1) * length - 1, "sequence"] = df[f"CH_{i+1}"][
                :length
            ].values
            temp.loc[i * length : (i + 1) * length - 1, "ID"] = ID
            temp.loc[i * length : (i + 1) * length - 1, "label"] = output_dict[ID]
    temp.dropna(inplace=True)
    temp_dfs.append(temp)
merged_df = pd.concat(temp_dfs).reset_index(drop=True)

X = merged_df[["sequence"]]
X_normalized = preprocessing.StandardScaler().fit_transform(X)
merged_df[["sequence"]] = X_normalized

# encode label :
merged_df["label"] = merged_df["label"].map({"F": 0, "M": 1})

merged_df.to_csv("test_df.csv")
