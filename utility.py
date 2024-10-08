import os
import shutil
import pandas as pd
import glob

# Load CSV file
df = pd.read_csv('FernSpores.csv')

# Base directory where image folders are located
base_dir = 'dataset/'

# Directories for the organized labels
organized_base_dir = 'organized/'
label_dirs = {
    'Trilete': os.path.join(organized_base_dir, 'Trilete'),
    'Monolete': os.path.join(organized_base_dir, 'Monolete')
}

# Create directories for labels if they don't exist
for dir_path in label_dirs.values():
    os.makedirs(dir_path, exist_ok=True)

# Find all image files under the dataset directory
image_paths = glob.glob(os.path.join(base_dir, '**', '*.jpg'), recursive=True)

# Create a dictionary for quick lookup of image paths
image_dict = {os.path.basename(path): path for path in image_paths}

# Move images to respective label directories
for index, row in df.iterrows():
    image_name = row['FileName']
    label = row['Label']

    src_path = image_dict.get(image_name)

    if src_path:
        dest_path = os.path.join(label_dirs[label], image_name)
        try:
            shutil.move(src_path, dest_path)
            print(f"Moved {src_path} to {dest_path}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
    else:
        print(f"File {image_name} not found in image dictionary.")
