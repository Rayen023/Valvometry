import pandas as pd
import numpy as np
import tqdm
import os
import logging
from sklearn.model_selection import StratifiedKFold
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F


# Logging
def get_log_file_name():
    counter = 1
    file_name = f"logs/train.log"
    while os.path.exists(file_name):
        counter += 1
        file_name = f"logs/train_{counter}.log"
    return file_name

logging.basicConfig(filename=get_log_file_name(), encoding='utf-8', level=logging.DEBUG)

# Hyperparameters
segments = 160
epochs = 500
batch_size = 32
num_layers = 3
lr = 0.001
weight_decay = 0.01
dropout = 0.2
hidden_size = 128
num_folds = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
try :
    X = torch.load('X1.t')
    y = torch.load('y1.t')
except :
    train_df = pd.read_csv("datasets/outputs/train_df_nt_after_norm.csv")
    #train_df = pd.read_csv("datasets/fordA.csv")
    train_df = train_df[~train_df['ID'].isin(['D1', 'A1', 'A4', 'C4', 'E3', 'F1', 'F4', 'G1', 'G3', 'G4', 'I2'])]
    logging.info(train_df.shape)

    # Create an empty DataFrame with the same column structure as train_df
    expanded_df = pd.DataFrame(columns=train_df.columns)

    # Iterating over each unique ID
    for unique_id in tqdm.tqdm(train_df['ID'].unique()):
        
        id_rows = train_df.loc[train_df['ID'] == unique_id]
        
        # Dividing the current ID into the desired number of segments
        segments = segments
        segment_length = len(id_rows) // segments
        
        for i in range(segments):
            segment_start = i * segment_length
            segment_end = (i + 1) * segment_length
            new_id = f"{unique_id}_{i+1}"
            new_rows = id_rows.iloc[segment_start:segment_end].copy()
            new_rows['ID'] = new_id
            expanded_df = pd.concat([expanded_df, new_rows])


    expanded_df.reset_index(drop=True, inplace=True)
    train_df = expanded_df.astype(train_df.dtypes)

    # Preprocess the data
    continuous_features = ['sequence']

    min_count = train_df.groupby("ID").size().min()
    print(f"Duree de chaque sequence = {min_count/36000} heures")

    sequences = []
    current_sequence = []

    for id in tqdm.tqdm(train_df['ID'].unique()) :
        current_sequence = train_df[train_df["ID"] == id][continuous_features].values
        sequences.append(current_sequence[:min_count])

    X = torch.tensor(np.array(sequences), dtype=torch.float64)
    y = torch.tensor(train_df.groupby('ID')['label'].first().values, dtype=torch.int64)

    torch.save(X, 'X1.t')
    torch.save(y, 'y1.t')

    X = torch.load('X1.t')
    y = torch.load('y1.t')


input_size = X.shape[2]
output_size = 1
print(f"input_size = {input_size}, output_size = {output_size}")

# Model

class RNNModel(nn.Module):
    def __init__(self):
        super(RNNModel, self).__init__()
        self.conv1 = nn.Conv1d(1, 128, kernel_size=16, padding=1)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=16, padding=1)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = nn.Conv1d(256, 64, kernel_size=16, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = F.relu(x)
        x = self.fc2(x)
        out = self.sigmoid(x)
        return out


# Cross Validation

dataset = TensorDataset(X, y)
StratifiedKFold = StratifiedKFold(n_splits=num_folds, shuffle=True)

best_loss = float('inf')  
best_model_path = 'best_model.pt'  



for fold, (train_indices, val_indices) in enumerate(StratifiedKFold.split(dataset , y)):

    model = RNNModel()
    model.to(device)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Create train and validation datasets based on the fold indices
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    # Create train and validation dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler = torch.utils.data.RandomSampler(train_dataset, replacement=False))
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)




    for epoch in range(epochs):
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        for inputs, targets in train_dataloader:
            model.train()
            
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            inputs = inputs.float()
            
            print(f"inputs_shape = {inputs.shape}")
            
            outputs = model(inputs)         
            
            # print outputs and targets
            print(f"outputs = {outputs}, targets = {targets}")
            outputs = outputs.squeeze()
            targets = targets.float().squeeze()
            
            # assert outputs.shape == targets.shape
            assert outputs.shape == targets.shape
            
            # Compute loss
            loss = criterion( outputs, targets )

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        with torch.no_grad():
            model.eval()
            for val_inputs, val_targets in val_dataloader:
                val_inputs = val_inputs.to(device).float()
                val_outputs = model(val_inputs).squeeze()
                
                val_targets = val_targets.to(device).float().squeeze()
                val_loss += criterion(val_outputs, val_targets).item()

                #logging.info(val_targets)
                #logging.info(val_outputs)
                
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
    logging.info(f'Accuracy: {accuracy * 100}%')
