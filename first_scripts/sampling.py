import pandas as pd
import numpy as np
import tqdm
import glob
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler


def reduce_mem_usage(props):
    start_mem_usg = props.memory_usage().sum() / 1024**2
    print("Memory usage of properties dataframe is :", start_mem_usg, " MB")
    NAlist = []  # Keeps track of columns that have missing values filled in.
    for col in props.columns:
        if props[col].dtype != object:  # Exclude strings

            # Print current column type
            print("******************************")
            print("Column: ", col)
            print("dtype before: ", props[col].dtype)

            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()

            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all():
                NAlist.append(col)
                props[col].fillna(mn - 1, inplace=True)

            # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = props[col] - asint
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)

            # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float32)

            # Print new column type
            print("dtype after: ", props[col].dtype)
            print("******************************")

    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024**2
    print("Memory usage is: ", mem_usg, " MB")
    print("This is ", 100 * mem_usg / start_mem_usg, "% of the initial size")
    return props


# Get genders dict
gender_csv = pd.read_excel(
    "/gpfs/scratch/rayen/Oysters/datasets/Original_Datasets/Individual sex_Microclosure counts_time closed.xlsx",
    header=1,
)
gender_dict = dict(zip(gender_csv.ID, gender_csv["Sex (histology)"]))
gender_dict

module = ["A", "C", "D", "E", "F", "G", "H", "I"]

temp_dfs = []

for letter in tqdm.tqdm(module):
    raw_data_files = sorted(
        glob.glob(
            f"/gpfs/scratch/rayen/Oysters/datasets/Original_Datasets/Raw valvo data files_2019-2020_Second experiment (Jeff)/{letter}*.csv"
        )
    )
    df = pd.concat(
        [
            pd.read_csv(raw_data_files[0], header=16, index_col=0),
            pd.read_csv(raw_data_files[1], header=16, index_col=0),
        ]
    ).reset_index(drop=True)
    df = reduce_mem_usage(df)
    length = df.shape[0]
    # length = length / 100
    length = int(length - (length % 60))
    print(length)
    temp = pd.DataFrame()
    temp["sequence"] = [np.nan for _ in range(length * 4)]
    temp["ID"] = [np.nan for _ in range(length * 4)]
    temp["label"] = [np.nan for _ in range(length * 4)]
    for i in range(4):
        ID = f"{letter}{i+1}"
        if ID in gender_dict:
            temp.loc[i * length : (i + 1) * length - 1, "sequence"] = df[f"CH_{i+1}"][
                :length
            ].values
            temp.loc[i * length : (i + 1) * length - 1, "ID"] = ID
            temp.loc[i * length : (i + 1) * length - 1, "label"] = gender_dict[ID]
    temp.dropna(inplace=True)
    temp_dfs.append(temp)
merged_df = pd.concat(temp_dfs).reset_index(drop=True)


# encode label :
merged_df["label"] = merged_df["label"].map({"F": 0, "M": 1})


def normalize_sq(group):
    X = group["sequence"]
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X.values.reshape(-1, 1))
    group["sequence"] = X_normalized
    return group


df_resampled = (
    merged_df.groupby("ID", group_keys=False).apply(normalize_sq).reset_index(drop=True)
)

df_resampled = reduce_mem_usage(df_resampled)

# df_resampled.sequence.plot(kind = "kde")

df_resampled.to_csv("train_df_nt_after_norm.csv")

print(df_resampled)
