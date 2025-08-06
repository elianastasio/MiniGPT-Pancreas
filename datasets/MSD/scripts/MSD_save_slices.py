import json
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from skimage import exposure

# Preprocess the image (histogram clipping and adjustment)
def preprocess_image(slice_data):
    lower, upper = -150, 250
    slice_data = np.clip(slice_data, lower, upper)
    slice_data = (slice_data - lower) / (upper - lower)
    slice_data = np.rot90(slice_data)      # Consistency with bboxes
    slice_data = np.fliplr(slice_data)
    return slice_data


# Load JSON data
def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

# Process slices: load volume, extract and preprocess each slice, then save as PNG
def process_slices(data, volumes_path, output_dir, balanced):
    previous_volume_name = None
    volume_data = None
    
    for entry in tqdm(data, desc='MSD'):
        volume_name = entry['volume_name']
        slice_index = entry['slice_index']

        # Load NIfTI volume only when switching to a new volume
        if volume_name != previous_volume_name:
            nifti_file = os.path.join(volumes_path, volume_name)
            nifti = nib.load(nifti_file)
            volume_data = nifti.get_fdata()
            previous_volume_name = volume_name

        # Extract 2D slice from 3D volume
        slice_data = volume_data[:, :, slice_index]

        # Remove .nii or .nii.gz extension from volume_name
        volume_name_without_extension = os.path.splitext(os.path.splitext(volume_name)[0])[0]

        output_filename = f"{volume_name_without_extension}_slice_{slice_index}.png"
        output_path = os.path.join(output_dir, output_filename)

        # If running --balanced after having run the base script, skip pancreas slices
        if os.path.exists(output_path) and balanced:
            # print(f'skipping slice {output_path}')
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
    volumes_path = '../raw_dataset/imagesTr'
    annotations_dir = '../annotations'
    prefix = 'MSD_balanced_' if args.balanced else 'MSD_'

    # Load train and test annotations
    train_data = load_json(os.path.join(annotations_dir, f"{prefix}train.json"))
    test_data = load_json(os.path.join(annotations_dir, f"{prefix}test.json"))
    combined_data = train_data + test_data

    # Process and save slices
    process_slices(combined_data, volumes_path, output_dir, args.balanced)
    print(f"MSD slices saved successfully in {output_dir}")
    print(f"Loaded {len(train_data)} train slices and {len(test_data)} test slices")
