import os
import gc
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset

import torchvision
from torchvision import transforms
from PIL import Image

print('Importing df')

train_df = pd.read_csv("/gpfs/scratch/rayen/Oysters/datasets/train_df2_after_norm_mem_reduce.csv")
print(train_df.groupby('ID').size())

print(f'train_df.shape with E_4 {train_df.shape}')
#train_df = train_df[~train_df['ID'].isin(['E_4',])]
print(f'train_df.shape without E_4 {train_df.shape}')


lr = 0.00051
num_epochs = 436
batch_size = 8
img_size = 276 
segment_hours = 8


def reduce_mem_usage(props):
    start_mem_usg = props.memory_usage().sum() / 1024**2
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in.
    for col in props.columns:
        if props[col].dtype != object:  # Exclude strings

            # Print current column type
            print("******************************")
            print("Column: ",col)
            print("dtype before: ",props[col].dtype)

            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()

            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all():
                NAlist.append(col)
                props[col].fillna(mn-1,inplace=True)

            # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = (props[col] - asint)
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
            print("dtype after: ",props[col].dtype)
            print("******************************")

    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024**2
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return props

train_df = reduce_mem_usage(train_df)
gc.collect()




# Filter out unwanted 'ID' values
#train_df = train_df[~train_df['ID'].isin(['D1', 'A1', 'A4', 'C4', 'E3', 'F1', 'F4', 'G1', 'G3', 'G4', 'I2'])]

def segment_group(group, segments):
    segment_length = len(group) // segments
    segmented_dfs = []

    for i in tqdm(range(segments)):
        segment_start = i * segment_length
        segment_end = (i + 1) * segment_length
        new_id = f"{group['ID'].iloc[0]}_{i+1}"
        new_rows = group.iloc[segment_start:segment_end].copy()
        new_rows['ID'] = new_id
        segmented_dfs.append(new_rows)

    return pd.concat(segmented_dfs)


nseconds = train_df[train_df['ID']=='A_2'].shape[0] /10
if int(nseconds) == 0 :
    nseconds = train_df[train_df['ID']=='A2'].shape[0] /10
total_hours = nseconds / 3600
total_days = total_hours / 24
nbr_segments = int(total_hours / segment_hours)
print(f'total_days : {total_days} total_hours : {total_hours}, segments : {nbr_segments}')
# Group by 'ID' and apply the segmentation function to each group
expanded_df = train_df.groupby('ID').apply(segment_group, nbr_segments)
expanded_df.reset_index(drop=True, inplace=True)
train_df = expanded_df.astype(train_df.dtypes)

min_count = train_df.groupby("ID").size().min()
print(f"Duree de chaque sequence = {min_count/36000} heures")

# Group the DataFrame by 'ID' and extract the values for each group
sequences = []
current_sequence = []
grouped = train_df.groupby('ID')
continuous_features = ['Sequence']
for name, group in tqdm(grouped):
    current_sequence = group[continuous_features].values
    sequences.append(current_sequence[:min_count])

X = np.array(np.array(sequences), dtype=np.float32)
y = np.array(train_df.groupby('ID')['label'].first().values, dtype=np.int8)

print(f"input_size = {X.shape}, output_size = {y.shape}")

def factors(n):
    result = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            result.append(i)
            result.append(n // i)
    return sorted(result)

factors_min_count = factors(min_count)  # Calculate the factors of min_count
middle_values = factors_min_count[len(factors_min_count) // 2 - 1:len(factors_min_count) // 2 + 1]  # Get the two middle values

print(factors_min_count)
print("Middle Values:", middle_values)  # Print the two middle values

images = [Image.fromarray(seq.reshape(*middle_values), 'L') for seq in X]

y = np.array(y).astype(int).squeeze()

# Split the images and labels into train and test sets.
X_train, X_test, y_train, y_test = train_test_split(images, y, test_size=0.2)
print(f"X_train shape: {len(X_train)}, X_test shape: {len(X_test)}, y_train shape: {y_train.shape}, y_test shape: {y_test.shape}")

print(type(X_train) , len(X_train))


class ImageDataset(Dataset):
    def __init__(self, images_series, labels, transform=None):
        self.images_series = images_series
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images_series)

    def __getitem__(self, idx):
        image = self.images_series[idx]

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]

        return image, label


test_transform = transforms.Compose([
    transforms.Resize((img_size,img_size)),
    transforms.ToTensor(),
    
])

train_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomCrop((144, 144)),
    #transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
])

train_dataset = ImageDataset(X_train, y_train,transform=train_transform)
test_dataset = ImageDataset(X_test, y_test,transform=test_transform)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    num_workers=4,
    shuffle=True
)
test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    num_workers=4,
    shuffle=True
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torchvision.models.efficientnet_b0()

model.features[0][0] = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
model.classifier.fc = nn.Linear(1000, 64)
model.classifier.fc1 = nn.Linear(64, 2)
model = model.to(device)

def train_model(model, train_dataloader, test_dataloader, criterion, optimizer, device, num_epochs=12):
    since = time.time()

    best_acc = 0.0
    best_epoch = 0

    for epoch in tqdm(range(num_epochs)):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Training phase
        model.train()
        running_train_loss = 0.0
        running_train_corrects = 0

        for inputs, labels in train_dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)

            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * inputs.size(0)
            running_train_corrects += torch.sum(preds == labels.data)

        train_loss = running_train_loss / len(train_dataloader.dataset)
        train_acc = running_train_corrects.double() / len(train_dataloader.dataset)

        print('Train Loss: {:.4f} Train Acc: {:.4f}'.format(train_loss, train_acc))

        # Testing phase
        model.eval()
        running_test_loss = 0.0
        running_test_corrects = 0

        with torch.no_grad():
            for inputs, labels in test_dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)

                running_test_loss += loss.item() * inputs.size(0)
                running_test_corrects += torch.sum(preds == labels.data)

        test_loss = running_test_loss / len(test_dataloader.dataset)
        test_acc = running_test_corrects.double() / len(test_dataloader.dataset)

        print('Test Loss: {:.4f} Test Acc: {:.4f}'.format(test_loss, test_acc))

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            #torch.save(model.state_dict(), os.path.join('/kaggle/working/', '{0:0=2d}.pth'.format(epoch+1)))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print(f'Best Acc: {best_acc},Best Epoch: {best_epoch} ')

    return model , best_acc ,  best_epoch


optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

# Setup the loss function
criterion = nn.CrossEntropyLoss()

# Train model
model , best_acc, best_epoch  = train_model(model, train_loader,test_loader, criterion, optimizer, device , num_epochs)

print(f"lr = {lr}")
print(f"num_epochs = {num_epochs}")
print(f"batch_size = {batch_size}")
print(f"img_size = {img_size}")
print(f"segment_hours = {segment_hours}")

transform_str = "\n".join([str(t) for t in train_transform.transforms])
print(transform_str)


# Read the existing DataFrame
df = pd.read_csv("/gpfs/scratch/rayen/Oysters/outputs.csv", index_col = None)

# Create a new DataFrame for the new values
new_row = pd.DataFrame({
    "lr": [lr],
    "num_epochs": [num_epochs],
    "batch_size": [batch_size],
    "img_size": [img_size],
    "segment_hours": [segment_hours],
    "Best Acc": [best_acc.cpu().numpy()],
    "Best Epoch": [best_epoch],
    "transforms": [transform_str]
})

# Concatenate the existing DataFrame with the new row
df = pd.concat([df, new_row], ignore_index=True)

# Save the updated DataFrame back to the CSV file
df.to_csv("/gpfs/scratch/rayen/Oysters/outputs.csv", index=False)
