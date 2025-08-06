import json
import os
import random
import argparse

def load_json(filename):
    with open(filename, 'r') as file:
        return json.load(file)

def save_json(data, filename):
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

# Split data into train and test sets based on volumes to avoid contamination between adjacent slices
def split_volumes(data, test_ratio=0.2, seed=42):
    
    # Group slices by volume
    random.seed(seed)
    volumes = {}
    for item in data:
        volume_name = item["volume_name"]
        if volume_name not in volumes:
            volumes[volume_name] = []
        volumes[volume_name].append(item)
    
    # Split volumes into train and test sets
    volume_names = list(volumes.keys())
    random.shuffle(volume_names)
    test_size = int(len(volume_names) * test_ratio)
    train_volume_names = volume_names[test_size:]
    test_volume_names = volume_names[:test_size]
    
    # separate data into train and test sets
    train_data = [item for volume in train_volume_names for item in volumes[volume]]
    test_data = [item for volume in test_volume_names for item in volumes[volume]]
    
    return train_data, test_data

def add_q_id(data):
    for i, item in enumerate(data, start=1):
        item['q_id'] = i
    return data

def balance_slices(data, threshold):
    volumes = {}
    for item in data:
        volume_name = item["volume_name"]
        if volume_name not in volumes:
            volumes[volume_name] = []
        volumes[volume_name].append(item)

    balanced_data = []
    for volume_name, slices in volumes.items():
        tumor_slices = [s for s in slices if s["tumor_bbox_ratio"] > threshold]
        non_tumor_slices = [s for s in slices if s["pixels_tumor"] == 0]
        if tumor_slices:
            count = len(tumor_slices)
            sampled_non_tumor = random.sample(non_tumor_slices, min(count, len(non_tumor_slices)))
            balanced_data.extend(tumor_slices + sampled_non_tumor)
        else:
            balanced_data.extend(tumor_slices)
    return balanced_data

# Process the datasets and generate split along various thresholds
def process_datasets(balanced=False):
    threshold = 0.6  # In the paper we use tumor_bbox_ratio > 0.6 to select valid tumor slices
    output_dir = "../annotations"

    data = load_json(os.path.join(output_dir, "MSD_slices_info.json"))
    train_data, test_data = split_volumes(data)
    
    filtered_train_data = [item for item in train_data if item["tumor_bbox_ratio"] > threshold]
    filtered_test_data = [item for item in test_data if item["tumor_bbox_ratio"] > threshold]
    
    if balanced: # Add non-tumor slices for balancing
        filtered_train_data = balance_slices(train_data, threshold)
        filtered_test_data = balance_slices(test_data, threshold)

    filtered_train_data = add_q_id(filtered_train_data)
    filtered_test_data = add_q_id(filtered_test_data)
    
    train_path = os.path.join(output_dir, f"MSD_tumor_{'balanced_' if balanced else ''}train.json")
    test_path = os.path.join(output_dir, f"MSD_tumor_{'balanced_' if balanced else ''}test.json")
    save_json(filtered_train_data, train_path)
    save_json(filtered_test_data, test_path)

    train_tumor = sum(1 for s in filtered_train_data if s["pixels_tumor"] > 0)
    train_non_tumor = len(filtered_train_data) - train_tumor
    test_tumor = sum(1 for s in filtered_test_data if s["pixels_tumor"] > 0)
    test_non_tumor = len(filtered_test_data) - test_tumor

    print(f"Train JSON saved at: {train_path} | Tumor: {train_tumor}, Non-tumor: {train_non_tumor}")
    print(f"Test JSON saved at: {test_path} | Tumor: {test_tumor}, Non-tumor: {test_non_tumor}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--balanced", action="store_true", help="Balance tumor and non-tumor slices per volume")
    args = parser.parse_args()
    process_datasets(balanced=args.balanced)