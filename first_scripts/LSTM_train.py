import pandas as pd
import numpy as np
import tqdm
import os
import logging
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# Logging
def get_log_file_name():
    counter = 1
    file_name = f"logs/train.log"
    while os.path.exists(file_name):
        counter += 1
        file_name = f"logs/train_{counter}.log"
    return file_name


logging.basicConfig(filename=get_log_file_name(), encoding="utf-8", level=logging.DEBUG)

# Hyperparameters
segments = 1
epochs = 500
batch_size = 32
num_layers = 50
lr = 0.01
weight_decay = 0.001
dropout = 0.2
hidden_size = 256
num_folds = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# train_df = pd.read_csv("datasets/outputs/train_df_nt_after_norm.csv")
train_df = pd.read_csv("datasets/fordA.csv")
train_df = train_df[
    ~train_df["ID"].isin(
        ["D1", "A1", "A4", "C4", "E3", "F1", "F4", "G1", "G3", "G4", "I2"]
    )
]
logging.info(train_df.shape)

# Create an empty DataFrame with the same column structure as train_df
expanded_df = pd.DataFrame(columns=train_df.columns)

# Iterating over each unique ID
for unique_id in tqdm.tqdm(train_df["ID"].unique()):

    id_rows = train_df.loc[train_df["ID"] == unique_id]

    # Dividing the current ID into the desired number of segments
    segments = segments
    segment_length = len(id_rows) // segments

    for i in range(segments):
        segment_start = i * segment_length
        segment_end = (i + 1) * segment_length
        new_id = f"{unique_id}_{i+1}"
        new_rows = id_rows.iloc[segment_start:segment_end].copy()
        new_rows["ID"] = new_id
        expanded_df = pd.concat([expanded_df, new_rows])


expanded_df.reset_index(drop=True, inplace=True)
train_df = expanded_df.astype(train_df.dtypes)

# Preprocess the data
continuous_features = ["sequence"]

min_count = train_df.groupby("ID").size().min()
print(f"Duree de chaque sequence = {min_count/36000} heures")

sequences = []
current_sequence = []

for id in tqdm.tqdm(train_df["ID"].unique()):
    current_sequence = train_df[train_df["ID"] == id][continuous_features].values
    sequences.append(current_sequence[:min_count])

X = torch.tensor(np.array(sequences), dtype=torch.float64)
y = torch.tensor(train_df.groupby("ID")["label"].first().values, dtype=torch.int64)

input_size = X.shape[2]
output_size = 1

# Model


class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn = nn.LSTM(
            input_size,
            hidden_size,
            batch_first=True,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.fc1 = nn.Linear(hidden_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, output_size)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(64)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()

        out, (hn, cn) = self.rnn(x, (h0.to(device), c0.to(device)))
        out = self.relu(self.fc1(out[:, -1, :]))
        out = self.bn1(out)
        out = self.relu(self.fc2(out))
        out = self.fc4(out)
        out = self.sigmoid(out)

        return out, (hn, cn)


# Cross Validation

dataset = TensorDataset(X, y)
StratifiedKFold = StratifiedKFold(n_splits=num_folds, shuffle=True)

best_loss = float("inf")
best_model_path = "best_model.pt"

for fold, (train_indices, val_indices) in enumerate(StratifiedKFold.split(dataset, y)):

    model = RNNModel(input_size, hidden_size, output_size, num_layers, dropout)
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Create train and validation datasets based on the fold indices
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    # Create train and validation dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    for epoch in range(epochs):
        for inputs, targets in train_dataloader:
            model.train()

            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            inputs = inputs.float()

            print(f"inputs_shape = {inputs.shape}")

            outputs, (hn, cn) = model(inputs)

            # print outputs and targets
            print(f"outputs = {outputs}, targets = {targets}")
            outputs = outputs.squeeze()

            # targets to float
            targets = targets.float().squeeze()

            # assert outputs.shape == targets.shape
            assert outputs.shape == targets.shape

            # Compute loss
            loss = criterion(outputs, targets)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        with torch.no_grad():
            model.eval()
            val_loss = 0.0
            for val_inputs, val_targets in val_dataloader:
                val_inputs = val_inputs.to(device).float()
                val_targets = val_targets.to(device)

                val_outputs, (hn, cn) = model(val_inputs)
                val_loss += criterion(
                    val_outputs.squeeze(), val_targets.float().squeeze()
                ).item()

                predicted = torch.round(val_outputs)

                val_total += val_targets.size(0)
                val_correct += (predicted == val_targets).sum().item()

            val_accuracy = val_correct / val_total
            val_loss /= len(val_dataloader)  # Average validation loss

            logging.info(
                f"Fold [{fold+1}/{num_folds}], Epoch [{epoch+1}/{epochs}], Training Loss: {loss.item()}, Validation Loss: {val_loss} , Validation Accuracy: {val_accuracy:.4f}"
            )

            # Save the model if there is an improvement in validation loss
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), best_model_path)
                logging.info("Saving the best model...")


# Load the best model for prediction
best_model = RNNModel(input_size, hidden_size, output_size)
best_model.load_state_dict(torch.load(best_model_path))

best_model = best_model.to("cpu")

with torch.no_grad():
    best_model.eval()

    X = X.to("cpu")

    outputs = best_model(X)
    logging.info(outputs)

    outputs = outputs.to("cpu").detach()

    predicted = torch.round(outputs.squeeze())
    accuracy = (predicted == y).sum().item() / len(y)
    logging.info(f"Accuracy: {accuracy * 100}%")
