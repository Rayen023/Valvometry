import pandas as pd
import numpy as np
import glob
from sklearn.preprocessing import LabelEncoder
import tqdm
import gc
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import KFold
import os
import logging

# Function to check if file exists and increment the file name
def get_log_file_name(base_name, extension):
    counter = 1
    file_name = f"logs/{base_name}.{extension}"
    while os.path.exists(file_name):
        counter += 1
        file_name = f"logs/{base_name}_{counter}.{extension}"
    return file_name

logging.basicConfig(filename=get_log_file_name('train', 'log'), encoding='utf-8', level=logging.DEBUG)


epochs = 500
batch_size = 12
num_layers = 4
lr = 0.001
weight_decay = 0.001
dropout = 0.2
hidden_size = 256
num_folds = 5



train_df = pd.read_csv("datasets/outputs/train_df_nt_after_norm.csv" , index_col = 0)
train_df = train_df[~train_df['ID'].isin(['D1', 'A1', 'A4', 'C4', 'E3', 'F1', 'F4', 'G1', 'G3', 'G4', 'I2'])]
logging.info(train_df.shape)

# Step 1: Create an empty DataFrame with the same column structure as train_df
expanded_df = pd.DataFrame(columns=train_df.columns)

# Step 2: Iterate over each unique ID
for unique_id in train_df['ID'].unique():
    # Step 3: Get the rows corresponding to the current ID
    id_rows = train_df.loc[train_df['ID'] == unique_id]
    
    # Step 4: Divide the current ID into the desired number of segments
    segments = 60

    segment_length = len(id_rows) // segments
    
    # Step 5: Create new rows by combining the divided IDs with the original gender and channel values
    for i in tqdm.tqdm(range(segments)):
        segment_start = i * segment_length
        segment_end = (i + 1) * segment_length
        new_id = f"{unique_id}_{i+1}"
        new_rows = id_rows.iloc[segment_start:segment_end].copy()
        new_rows['ID'] = new_id
        expanded_df = pd.concat([expanded_df, new_rows])

# Reset the index of the expanded DataFrame
expanded_df.reset_index(drop=True, inplace=True)

expanded_df = expanded_df.astype(train_df.dtypes)

# logging.info the expanded DataFrame
train_df = expanded_df

grouped = train_df.groupby("ID")

min_count = grouped.size().min()

# Preprocess the data
continuous_features = ['sequence']

# Encode gender labels
y = train_df.groupby('ID')['label'].first().values

IDs = train_df.ID.unique()

sequences = []
current_sequence = []

for id in tqdm.tqdm(IDs) :
    current_sequence = train_df[train_df["ID"] == id][continuous_features].values
    sequences.append(current_sequence[:min_count])

X = torch.tensor(np.array(sequences), dtype=torch.float64)
y = torch.tensor(y, dtype=torch.int64)

input_size = X.shape[1]
output_size = 1


best_loss = float('inf')  # Initialize the best loss as infinity
best_model_path = 'best_model.pt'  # Define the path to save the best model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class FeedForwardModel(nn.Module):
    def __init__(self, input_size, output_size, dropout):
        super(FeedForwardModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(64, 32)
        self.dropout3 = nn.Dropout(dropout)
        self.fc4 = nn.Linear(32, output_size)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.tanh(self.fc1(x.float()))
        out = self.dropout1(out)
        out = self.tanh(self.fc2(out))
        out = self.dropout2(out)
        out = self.tanh(self.fc3(out))
        out = self.dropout3(out)
        out = self.fc4(out)
        out = self.sigmoid(out)
        
        # Squeeze the output tensor to match target tensor's dimensions
        out = out.squeeze()

        return out

model = FeedForwardModel(input_size, output_size, dropout)
model.to(device)

# Define the loss function and optimizer
criterion = nn.BCELoss()  # Use BCELoss instead of BCEWithLogitsLoss for binary classification
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


logging.info(model)
logging.info(len(list(model.parameters())))
for i in range(len(list(model.parameters()))):
    logging.info(list(model.parameters())[i].size())



dataset = TensorDataset(X, y)

kfold = KFold(n_splits=num_folds, shuffle=False)

for fold, (train_indices, val_indices) in enumerate(kfold.split(dataset)):
    # Create train and validation datasets based on the fold indices
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    # Create train and validation dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    for epoch in range(epochs):
        
        for inputs, targets in train_dataloader:
            
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(inputs.squeeze())

            # Compute loss
            loss = criterion(outputs.view(len(targets), 1), targets.view(len(targets), 1).float())

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            for val_inputs, val_targets in val_dataloader:
                
                val_inputs = val_inputs.to(device)
                val_targets = val_targets.to(device)

                logging.info(val_inputs.shape)
                val_outputs = model(val_inputs.squeeze())
                val_loss += criterion(val_outputs.view(len(val_targets), 1), val_targets.view(len(val_targets), 1).float()).item()
                
                predicted = torch.round(val_outputs)
                
                val_total += val_targets.size(0)
                val_correct += (predicted == val_targets).sum().item()

            val_accuracy = val_correct / val_total
            val_loss /= len(val_dataloader)  # Average validation loss

            logging.info(f'Fold [{fold+1}/{num_folds}], Epoch [{epoch+1}/{epochs}], Training Loss: {loss.item()}, Validation Loss: {val_loss} , Validation Accuracy: {val_accuracy:.4f}')

            # Save the model if there is an improvement in validation loss
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            logging.info("Saving the best model...")

        model = model.to(device)
        model.train()
