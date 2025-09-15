import json
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from collections import defaultdict

# Preprocess the image (histogram clipping and adjustment)
def preprocess_image(slice_data):
    lower, upper = -150, 250
    slice_data = np.clip(slice_data, lower, upper)
    slice_data = (slice_data - lower) / (upper - lower)
    slice_data = np.rot90(slice_data)      # Consistency with bboxes
    slice_data = np.fliplr(slice_data)
    return slice_data

def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def get_volume_path(volume_name):
    """Return the full path to the volume based on the Part it belongs to."""
    case_id = int(volume_name.split('_')[1])
    if 1 <= case_id <= 400:
        part_dir = '../raw_dataset/AbdomenCT-1K-ImagePart1'
    elif 401 <= case_id <= 800:
        part_dir = '../raw_dataset/AbdomenCT-1K-ImagePart2'
    else:
        part_dir = '../raw_dataset/AbdomenCT-1K-ImagePart3'
    return os.path.join(part_dir, volume_name)

def process_slices(data, output_dir):
    # Group entries by volume
    volumes_dict = defaultdict(list)
    for entry in data:
        volumes_dict[entry['volume_name']].append(entry)

    for volume_name, entries in tqdm(volumes_dict.items(), desc='Processing volumes'):
        volume_name_no_ext = os.path.splitext(os.path.splitext(volume_name)[0])[0]

        # Load the volume once per case
        nifti_file = get_volume_path(volume_name)
        if not os.path.exists(nifti_file):
            print(f"Warning: Volume {nifti_file} not found, skipping...")
            continue

        nifti = nib.load(nifti_file)
        volume_data = nifti.get_fdata()

        # Collect unique slice indices
        slice_indices = sorted({entry['slice_index'] for entry in entries})

        # Process and save each slice
        for slice_index in slice_indices:
            output_filename = f"{volume_name_no_ext}_slice_{slice_index}.png"
            output_path = os.path.join(output_dir, output_filename)

            # Skip if already saved
            if os.path.exists(output_path):
                continue

            # Extract and process slice
            slice_data = volume_data[:, :, slice_index]
            processed_slice = preprocess_image(slice_data)

            # Save to disk
            plt.imsave(output_path, processed_slice, cmap='gray')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json", type=str, default="../annotations/ABD_train.json")
    parser.add_argument("--test_json", type=str, default="../annotations/ABD_test.json")
    parser.add_argument("--output_dir", type=str, default="../slices")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load train and test annotations
    train_data = load_json(args.train_json)
    test_data = load_json(args.test_json)
    combined_data = train_data + test_data

    process_slices(combined_data, args.output_dir)
    print(f"ABD slices saved successfully in {args.output_dir}")
    print(f"Loaded {len(train_data)} train slices and {len(test_data)} test slices")
