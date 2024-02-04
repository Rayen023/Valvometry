import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from statistics import mean, stdev
from sklearn.mixture import GaussianMixture
from tqdm import tqdm
from sklearn.metrics import accuracy_score

# Read the CSV file
train_df = pd.read_csv("datasets/outputs/train_df_nt_after_norm.csv", index_col=0)
print(train_df)

# drop rows with ID D1 or F1
train_df = train_df[
    ~train_df["ID"].isin(
        ["D1", "A1", "A4", "C4", "E3", "F1", "F4", "G1", "G3", "G4", "I2"]
    )
]


# Extract labels


# Step 1: Create an empty DataFrame with the same column structure as train_df
expanded_df = pd.DataFrame(columns=train_df.columns)

# Step 2: Iterate over each unique ID and create new rows
for unique_id in tqdm(train_df["ID"].unique()):
    id_rows = train_df[train_df["ID"] == unique_id]

    # Divide the current ID into the desired number of segments
    segments = 1
    segment_length = len(id_rows) // segments

    # Create new rows by combining the divided IDs with the original gender and channel values
    for i in range(segments):
        segment_start = i * segment_length
        segment_end = (i + 1) * segment_length
        new_id = f"{unique_id}_{i+1}"
        new_rows = id_rows.iloc[segment_start:segment_end].copy()
        new_rows["ID"] = new_id
        new_rows["first_derivative"] = id_rows.sequence.diff()
        new_rows["second_derivative"] = new_rows["first_derivative"].diff()
        expanded_df = pd.concat([expanded_df, new_rows])

# Reset the index of the expanded DataFrame
expanded_df.reset_index(drop=True, inplace=True)

# Drop rows with missing values
train_df = expanded_df.dropna()

continuous_features = ["sequence", "first_derivative", "second_derivative"]

min_count = train_df.groupby("ID").size().min()

print(
    "Moyenne label : ",
    train_df.groupby("label")[
        ["sequence", "first_derivative", "second_derivative"]
    ].mean(),
)
print(
    "Std : ",
    train_df.groupby("label")[
        ["sequence", "first_derivative", "second_derivative"]
    ].std(),
)
print(
    "Moyenne ID: ",
    train_df.groupby("ID")[
        ["sequence", "first_derivative", "second_derivative"]
    ].mean(),
)
print(
    "Std : ",
    train_df.groupby("ID")[["sequence", "first_derivative", "second_derivative"]].std(),
)

# min_count= min_count // 17

# min_count = (min_count // 4) * 4


# Get sequences and labels
sequences = []
IDs = train_df.ID.unique()
y = train_df.groupby("ID")["label"].first().values

ys = []

for id in tqdm(IDs):
    current_sequence = train_df[train_df["ID"] == id][continuous_features].values
    sequences.append(current_sequence[-min_count:])
    current_y = train_df[train_df["ID"] == id]["label"].values
    ys.append(current_y[-min_count:])

sequences = np.array(sequences)
y = np.array(y).squeeze()
# print("before" , sequences[0])

# section_length = int(min_count/4)

# selected_sequences = sequences.reshape(len(sequences),4,section_length,3)[:,3,:, :]
# sequences = selected_sequences
# print("after" , sequences[0])

# ys = np.array(ys).squeeze()

# ys = ys.reshape(-1,1).squeeze()

# ys = ys[3::4]

# Create new dataframe
# new_df = pd.DataFrame(columns=['label', 'sequence', 'first_derivative', 'second_derivative'])
# new_df['label'] = ys
# new_df[['sequence', 'first_derivative', 'second_derivative']] = sequences.reshape(-1, 3)

# Save the new dataframe
# new_df.to_csv("datasets/outputs/train_df_nt_after_norm_7.csv")

print(sequences.shape)

nsamples, nx, ny = sequences.shape
sequences = sequences.reshape((nsamples, nx * ny))

X_train = sequences
y_train = y.astype(int)

n_components = 2
gm = GaussianMixture(
    n_components=n_components,
    covariance_type="diag",
    n_init=32,
    random_state=0,
    max_iter=250,
)

skf = StratifiedKFold(n_splits=5, shuffle=True)

lst_accu_stratified = []

print(y_train)

for train_index, test_index in tqdm(skf.split(X_train, y_train)):
    x_train_fold, x_test_fold = X_train[train_index], X_train[test_index]
    y_train_fold, y_test_fold = y_train[train_index], y_train[test_index]

    gm.fit(x_train_fold.squeeze(), y_train_fold.squeeze())
    lst_accu_stratified.append(gm.score(x_test_fold.squeeze(), y_test_fold.squeeze()))

# Print the output
print("List of possible accuracy:", lst_accu_stratified)
print(
    "\nMaximum Accuracy That can be obtained from this model is:",
    max(lst_accu_stratified) * 100,
    "%",
)
print("\nMinimum Accuracy:", min(lst_accu_stratified) * 100, "%")
print("\nOverall Accuracy:", mean(lst_accu_stratified) * 100, "%")
print("\nStandard Deviation is:", stdev(lst_accu_stratified))
preds = gm.predict(sequences)
print(accuracy_score(y, preds))
print(gm.n_features_in_, gm.weights_)
