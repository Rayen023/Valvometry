import pandas as pd
import os
import glob


# Define a function to extract information from each file
def extract_info_from_file(file_path):
    info = {}
    with open(file_path, "r") as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith("Best Acc:"):
                # Extract Best Acc and Best Epoch values
                parts = line.strip().split(",")
                for part in parts:
                    key, value = part.split(": ")
                    info[key] = float(value)
            elif line.startswith("lr ="):
                info["lr"] = float(line.split("=")[1].strip())
            elif line.startswith("num_epochs ="):
                info["num_epochs"] = int(line.split("=")[1].strip())
            elif line.startswith("batch_size ="):
                info["batch_size"] = int(line.split("=")[1].strip())
            elif line.startswith("img_size ="):
                info["img_size"] = int(line.split("=")[1].strip())
            elif line.startswith("segment_hours ="):
                info["segment_hours"] = int(line.split("=")[1].strip())
    return info


# Directory containing the files
directory = "/gpfs/scratch/rayen/Oysters/output/"  # Replace with your directory path

# Use glob to find all files ending with .out
file_paths = glob.glob(os.path.join(directory, "*.out"))

# Initialize a list to store extracted information
info_list = []

# Iterate through each file and extract information
for file_path in file_paths:
    info = extract_info_from_file(file_path)
    info_list.append(info)

# Create a DataFrame from the list of dictionaries
df = pd.DataFrame(info_list)
df.to_csv(os.path.join(directory, "outputs.csv"))
# Print the resulting DataFrame
print(df)
