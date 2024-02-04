import gc
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
from PIL import Image
import re
import random
import os
import optuna

from datetime import datetime

timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
save_directory = "best_{0}".format(timestamp)
os.makedirs(save_directory, exist_ok=True)

import warnings

# Suppress specific UserWarning from the torch DataLoader
warnings.filterwarnings(
    "ignore", category=UserWarning, module="torch.utils.data.dataloader"
)


gc.collect()


train_df = pd.read_csv("datasets/train_df2_after_norm_mem_reduce.csv")
train_df = train_df[
    ~train_df["ID"].isin(
        [
            "A_1",
            "B_3",
            "B_4",
            "E_4",
        ]
    )
]

# test_df = train_df.copy()

test_df = pd.read_csv("datasets/train_df_nt_after_norm_mem_reduce.csv")
test_df = test_df[
    ~test_df["ID"].isin(
        [
            "A1",
            "A4",
            "C4",
            "D1",
            "E3",
            "F1",
            "G1",
            "G4",
        ]
    )
]


print(train_df.head())
print(test_df.head())

continuous_features = "Sequence"
train_continuous_features = [
    col for col in train_df.columns if col.lower() == continuous_features.lower()
]
test_continuous_features = [
    col for col in test_df.columns if col.lower() == continuous_features.lower()
]

lr = 0.000508
num_epochs = 100
batch_size = 8
img_size = 276
segment_hours = 8
crop_size = 144


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


train_df = reduce_mem_usage(train_df)
test_df = reduce_mem_usage(test_df)

gc.collect()


# Define a function to segment the data within each group
def segment_group(group, segments):
    segment_length = len(group) // segments
    segmented_dfs = []

    for i in tqdm(range(segments)):
        segment_start = i * segment_length
        segment_end = (i + 1) * segment_length
        new_id = f"{group['ID'].iloc[0]}_{i+1}"
        new_rows = group.iloc[segment_start:segment_end].copy()
        new_rows["ID"] = new_id
        segmented_dfs.append(new_rows)

    return pd.concat(segmented_dfs)


def sequence_df(df, hours, continuous_features):

    nseconds = df["ID"].value_counts().get("D3", 0) / 10
    if int(nseconds) == 0:
        nseconds = df["ID"].value_counts().get("D_3", 0) / 10
    total_hours = nseconds / 3600
    total_days = total_hours / 24
    nbr_segments = int(total_hours / hours)
    print(
        f"total_days : {total_days} total_hours : {total_hours}, segments : {nbr_segments}"
    )

    expanded_df = df.groupby("ID").apply(segment_group, nbr_segments)

    print("Reseting the index and converting types")

    expanded_df.reset_index(drop=True, inplace=True)
    df = expanded_df.astype(df.dtypes)

    print("Calculating min count")

    min_count = df.groupby("ID").size().min()
    print(f"Min_count {min_count} Duree de chaque sequence = {min_count/36000} heures")

    grouped = df.groupby("ID")

    def natural_sort_key(s):
        return [
            int(text) if text.isdigit() else text.lower()
            for text in re.split(r"(\d+)", s)
        ]

    sequences = []
    ids = []

    for name in sorted(grouped.groups.keys(), key=natural_sort_key):
        group = grouped.get_group(name)
        current_sequence = group[continuous_features].values
        sequences.append(current_sequence[:min_count])
        ids.append(name)

    # Using index splitting for train and test

    X = np.array(np.array(sequences), dtype=np.float32)
    y = np.array(df.groupby("ID")["label"].first().values, dtype=int)

    y = np.array(y, dtype=int).squeeze()

    print(f"input_size = {X.shape}, output_size = {y.shape}")

    return X.squeeze(), y.squeeze(), min_count, ids, nbr_segments


X, y, min_count, train_ids, nbr_segments = sequence_df(
    train_df, segment_hours, train_continuous_features
)
X_test, y_test, test_min_count, test_ids, test_nbr_segments = sequence_df(
    test_df, segment_hours, test_continuous_features
)

min_count = min(min_count, test_min_count)

print(f"input_size = {X.shape}, output_size = {y.shape}")
print(f"input_size = {X_test.shape}, output_size = {y_test.shape}")

X = X[:, :min_count]
X_test = X_test[:, :min_count]

print(f"input_size = {X.shape}, output_size = {y.shape}")
print(f"input_size = {X_test.shape}, output_size = {y_test.shape}")

print(len(test_ids))


def factors(n):
    result = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            result.append(i)
            result.append(n // i)
    return sorted(result)


factors_min_count = factors(min_count)
middle_values = factors_min_count[
    len(factors_min_count) // 2 - 1 : len(factors_min_count) // 2 + 1
]  # Get the two middle values
print(factors_min_count)
print("Middle Values:", middle_values)


class ImageDataset(Dataset):
    def __init__(self, images_series, labels, ids, transform=None):
        self.images_series = images_series
        self.labels = labels
        self.ids = ids
        self.transform = transform

    def __len__(self):
        return len(self.images_series)

    def __getitem__(self, idx):
        image = self.images_series[idx]

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]

        id = self.ids[idx]

        return image, label, id


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Y = y[::nbr_segments]
train_ids = np.array(train_ids)

Y_test = y_test[::test_nbr_segments]
test_ids = np.array(test_ids)


def train_model(
    model, train_dataloader, test_dataloader, criterion, optimizer, device, num_epochs
):
    since = time.time()

    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []

    best_acc = 0.0
    best_epoch = 0

    for epoch in tqdm(range(num_epochs)):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # Training phase
        model.train()
        running_train_loss = 0.0
        running_train_corrects = 0

        for inputs, labels, _ in train_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            outputs = outputs.squeeze()
            labels = labels.float().view_as(outputs)
            loss = criterion(outputs, labels.float())

            preds = (torch.sigmoid(outputs) > 0.5).float()

            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * inputs.size(0)
            running_train_corrects += torch.sum(preds == labels.data)

        train_loss = running_train_loss / len(train_dataloader.dataset)
        train_acc = running_train_corrects.double() / len(train_dataloader.dataset)

        print("Train Loss: {:.4f} Train Acc: {:.4f}".format(train_loss, train_acc))

        train_losses.append(train_loss)
        train_accuracies.append(train_acc.item())

        # Testing phase
        model.eval()
        running_test_loss = 0.0
        running_test_corrects = 0

        columns = [
            "labels",
            "segment_ids",
            "ids",
            "outputs",
            "sigmoid_outputs",
            "preds",
            "pair_best_epoch",
            "pair_best_acc",
        ]
        results_df = pd.DataFrame(columns=columns)

        with torch.no_grad():
            for inputs, labels, ids in test_dataloader:
                # print(inputs.shape)
                # print(inputs)
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                outputs = outputs.squeeze()

                labels = labels.float().view_as(outputs)
                loss = criterion(outputs, labels.float())

                preds = (torch.sigmoid(outputs) > 0.5).float()
                # print(preds)
                # print('preds', preds)

                running_test_loss += loss.item() * inputs.size(0)
                running_test_corrects += torch.sum(preds == labels.data)

                batch_results = pd.DataFrame(
                    {
                        "labels": labels.cpu().numpy(),
                        "segment_ids": [segment_id[:3] for segment_id in ids],
                        "ids": ids,
                        "outputs": outputs.cpu().numpy(),
                        "sigmoid_outputs": torch.sigmoid(outputs).cpu().numpy(),
                        "preds": preds.cpu().numpy(),
                        "pair_best_epoch": 0,
                        "pair_best_acc": 0,
                    }
                )

                results_df = pd.concat([results_df, batch_results], ignore_index=True)

            test_loss = running_test_loss / len(test_dataloader.dataset)
            test_acc = running_test_corrects.double() / len(test_dataloader.dataset)

            print("Test Loss: {:.4f} Test Acc: {:.4f}".format(test_loss, test_acc))

            test_losses.append(test_loss)
            test_accuracies.append(test_acc.item())

            if test_acc > best_acc and epoch > 50:
                best_acc = test_acc.item()
                best_epoch = epoch
                best_results_df = results_df.copy()
                best_results_df["pair_best_acc"] = best_acc
                best_results_df["pair_best_epoch"] = best_epoch

                torch.save(
                    model.state_dict(),
                    os.path.join(save_directory, "{:02d}.pth".format(epoch + 1)),
                )

    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print(f"Best Acc: {best_acc}, Best Epoch: {best_epoch}")

    return model, best_acc, best_epoch, best_results_df


columns = [
    "labels",
    "segment_ids",
    "ids",
    "outputs",
    "sigmoid_outputs",
    "preds",
    "pair_best_epoch",
    "pair_best_acc",
]
all_results_df = pd.DataFrame(columns=columns)

X_train = X
y_train = y

Train_images = [seq.reshape(*middle_values) for seq in X_train]
Test_images = [seq.reshape(*middle_values) for seq in X_test]

print(
    f"X_train shape: {len(X_train)}, X_test shape: {len(X_test)}, y_train shape: {y_train.shape}, y_test shape: {y_test.shape}"
)

train_transform = transforms.Compose(
    [
        # transforms.Resize((img_size, img_size)),
        # transforms.RandomCrop((crop_size, crop_size)),
        transforms.ToTensor(),
    ]
)

test_transform = transforms.Compose(
    [
        # transforms.Resize((img_size, img_size)),
        # transforms.RandomCrop((crop_size, crop_size)),
        transforms.ToTensor(),
    ]
)

transform_str = "\n".join([str(t) for t in train_transform.transforms])
print(transform_str)

print(f"lr = {lr}")
print(f"num_epochs = {num_epochs}")
print(f"batch_size = {batch_size}")
print(f"img_size = {img_size}")
print(f"segment_hours = {segment_hours}")
if crop_size:
    print(f"crop_size = {crop_size}")


# original_train_dataset = ImageDataset(X_train, y_train,train_ids, transform=test_transform)
train_dataset = ImageDataset(
    Train_images, y_train, train_ids, transform=train_transform
)

# train_dataset = torch.utils.data.ConcatDataset([train_dataset, original_train_dataset])
test_dataset = ImageDataset(Test_images, y_test, test_ids, transform=test_transform)


def objective(trial):

    param = {
        "batch_size": trial.suggest_int(
            "batch_size", 8, 32
        ),  # Adjusted batch size interval
        "lr": trial.suggest_float(
            "lr", 0.0001, 0.001
        ),  # Adjusted learning rate interval
        "epochs": trial.suggest_int(
            "epochs", 30, 80
        ),  # Adjusted number of epochs interval
    }

    batch_size = param["batch_size"]
    lr = param["lr"]
    num_epochs = param["epochs"]

    print(f"batch_size = {batch_size}")
    print(f"lr = {lr}")
    print(f"num_epochs = {num_epochs}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, num_workers=4, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, num_workers=4, shuffle=False
    )

    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()

            # Convolutional layers
            self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
            self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

            # Max pooling layer
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

            # Fully connected layers
            self.fc1 = nn.Linear(
                256 * (514 // 8) * (563 // 8), 512
            )  # Adjust based on your image dimensions
            self.fc2 = nn.Linear(512, 1)

            # Dropout for regularization
            self.dropout = nn.Dropout(0.5)  # You can adjust the dropout rate

            # Batch normalization
            self.batch_norm1 = nn.BatchNorm2d(64)
            self.batch_norm2 = nn.BatchNorm2d(128)
            self.batch_norm3 = nn.BatchNorm2d(256)

        def forward(self, x):
            x = self.batch_norm1(torch.relu(self.conv1(x)))
            x = self.pool(x)

            x = self.batch_norm2(torch.relu(self.conv2(x)))
            x = self.pool(x)

            x = self.batch_norm3(torch.relu(self.conv3(x)))
            x = self.pool(x)

            # Flatten before fully connected layers
            x = x.view(
                -1, 256 * (514 // 8) * (563 // 8)
            )  # Adjust based on your image dimensions

            x = self.dropout(torch.relu(self.fc1(x)))
            x = self.fc2(x)

            return x

    # Instantiate the model
    model = SimpleCNN()

    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    criterion = nn.BCEWithLogitsLoss()

    # Train model
    model, best_acc, best_epoch, best_results_df = train_model(
        model, train_loader, test_loader, criterion, optimizer, device, num_epochs
    )

    return best_acc


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

print("Number of finished trials:", len(study.trials), study.best_trial.values)
print("Best trial:", study.best_trial.params)


all_results_df = pd.concat([all_results_df, all_results_df], ignore_index=True)
correct_predictions_df = all_results_df[
    all_results_df["labels"] == all_results_df["preds"]
]
correct_predictions_count_by_segment = (
    correct_predictions_df.groupby("segment_ids")
    .size()
    .reset_index(name="correct_segments_count")
)
correct_predictions_count_by_segment["ID_acc"] = (
    correct_predictions_count_by_segment["correct_segments_count"] / test_nbr_segments
)
numeric_columns = [
    "labels",
    "outputs",
    "sigmoid_outputs",
    "preds",
    "pair_best_epoch",
    "pair_best_acc",
]
all_results_df[numeric_columns] = all_results_df[numeric_columns].apply(
    pd.to_numeric, errors="coerce"
)
all_results_df["segment_ids"] = pd.Categorical(
    all_results_df["segment_ids"],
    categories=all_results_df["segment_ids"].unique(),
    ordered=True,
)
grouped_results = (
    all_results_df.groupby("segment_ids")
    .agg(
        {
            "labels": "mean",
            "outputs": "mean",
            "sigmoid_outputs": ["mean", "std"],
            "preds": ["mean", "std"],
            "pair_best_epoch": "mean",
            "pair_best_acc": "mean",
        }
    )
    .reset_index()
)
grouped_results.columns = [
    "segment_ids",
    "labels_mean",
    "outputs_mean",
    "sigmoid_outputs_mean",
    "sigmoid_outputs_std",
    "preds_mean",
    "preds_std",
    "pair_best_epoch_mean",
    "pair_best_acc_mean",
]
merged_df = pd.merge(
    grouped_results, correct_predictions_count_by_segment, on="segment_ids", how="left"
)
print(all_results_df)
print(merged_df)
if crop_size:
    folder_name_base = f"efficientnet_b0_lr_{lr}_epochs_{num_epochs}_batch_{batch_size}_img_{img_size}_crop_{crop_size}_segments_{segment_hours}h"
else:
    folder_name_base = f"efficientnet_b0_lr_{lr}_epochs_{num_epochs}_batch_{batch_size}_img_{img_size}_segments_{segment_hours}h"
folder_name = folder_name_base
counter = 1
while os.path.exists(folder_name):
    folder_name = f"{folder_name_base}{counter}"
    counter += 1
os.makedirs(folder_name, exist_ok=True)
merged_df.to_csv(os.path.join(folder_name, "merged_df.csv"), index=False)
all_results_df.to_csv(os.path.join(folder_name, "all_results_df.csv"), index=False)
