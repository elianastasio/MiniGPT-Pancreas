import json
import os
import numpy as np
import pydicom
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

# Preprocess the image (histogram clipping and adjustment)
def preprocess_image(slice_data):
    lower, upper = -150, 250
    slice_data = np.clip(slice_data, lower, upper)
    slice_data = (slice_data - lower) / (upper - lower)
    slice_data = np.rot90(slice_data, 2)  # Consistency with bboxes
    slice_data = np.fliplr(slice_data) # The volume slices have different orientation than the label volumes
    return slice_data

# Load JSON data
def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

# Process slices: load DICOM slices from folders, preprocess, and save as PNG
def process_slices(data, volumes_path, output_dir, balanced):
    for entry in tqdm(data, desc='NIH'):
        volume_name = entry['volume_name']
        volume_index = volume_name[5:9]
        volume_folder_name = "PANCREAS_" + volume_index
        slice_index = entry['slice_index']

        # Build the path to the DICOM folder (2 levels down)
        volume_folder = os.path.join(volumes_path, volume_folder_name)
        level_1 = os.listdir(volume_folder)[0]  # First subfolder
        level_2 = os.listdir(os.path.join(volume_folder, level_1))[0]  # Second subfolder
        dicom_folder = os.path.join(volume_folder, level_1, level_2)

        # Build the DICOM file path
        dicom_file = os.path.join(dicom_folder, f"1-{slice_index + 1:03d}.dcm")  # DICOMs are 1-indexed
        if not os.path.exists(dicom_file):
            print(f"Warning: missing file {dicom_file}")
            continue

        # Read DICOM
        dcm = pydicom.dcmread(dicom_file)
        slice_data = dcm.pixel_array.astype(np.float32)

        # Build output path
        volume_name_without_extension = volume_folder_name
        output_filename = f"{volume_name_without_extension}_slice_{slice_index}.png"
        output_path = os.path.join(output_dir, output_filename)

        # Skip if balanced mode and file exists
        if os.path.exists(output_path) and balanced:
            print(f'skipping slice {output_path}')
            continue

        # Preprocess slice
        processed_slice = preprocess_image(slice_data)

        # Save slice
        plt.imsave(output_path, processed_slice, cmap='gray')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--balanced", action="store_true", help="Use balanced JSONs")
    args = parser.parse_args()

    # Define paths
    output_dir = '../slices'
    os.makedirs(output_dir, exist_ok=True)
    volumes_path = '../raw_dataset/volumes'
    annotations_dir = '../annotations'
    prefix = 'NIH_balanced_' if args.balanced else 'NIH_'

    # Load train and test annotations
    train_data = load_json(os.path.join(annotations_dir, f"{prefix}train.json"))
    test_data = load_json(os.path.join(annotations_dir, f"{prefix}test.json"))
    combined_data = train_data + test_data

    # Process and save slices
    process_slices(combined_data, volumes_path, output_dir, args.balanced)
    print(f"NIH slices saved successfully in {output_dir}")
    print(f"Loaded {len(train_data)} train slices and {len(test_data)} test slices")
