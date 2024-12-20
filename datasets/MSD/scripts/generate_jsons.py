import json
import os
import random

# load json
def load_json(filename):
    with open(filename, 'r') as file:
        return json.load(file)

# save json
def save_json(data, filename):
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

# split data into train and test sets based on volumes
def split_volumes(data, test_ratio=0.2, seed=42):
    
    # group slices by volume
    random.seed(seed)
    volumes = {}
    for item in data:
        volume_name = item["volume_name"]
        if volume_name not in volumes:
            volumes[volume_name] = []
        volumes[volume_name].append(item)
    
    # split volumes into train and test sets
    volume_names = list(volumes.keys())
    random.shuffle(volume_names)
    test_size = int(len(volume_names) * test_ratio)
    train_volume_names = volume_names[test_size:]
    test_volume_names = volume_names[:test_size]
    
    # separate data into train and test sets
    train_data = [item for volume in train_volume_names for item in volumes[volume]]
    test_data = [item for volume in test_volume_names for item in volumes[volume]]
    
    return train_data, test_data

# add unique q_id to each item in the data
def add_q_id(data):
    for i, item in enumerate(data, start=1):
        item['q_id'] = i
    return data

# main func: process the datasets and generate splits along various thresholds
def process_datasets():
    datasets = ["MSD"]
    thresholds = [i / 10 for i in range(10)]
    output_dir = "../annotations"
    summary = {}

    for dataset in datasets:
        # load dataset info
        data = load_json(os.path.join(output_dir, "slices_info.json"))
        
        # split the data into train and test sets based on volumes
        train_data, test_data = split_volumes(data)
        
        for threshold in thresholds:
            # filter data based on threshold
            filtered_train_data = [item for item in train_data if item["pancreas_bbox_ratio"] > threshold]
            filtered_test_data = [item for item in test_data if item["pancreas_bbox_ratio"] > threshold]
            
            # add q_id to each annotation
            filtered_train_data = add_q_id(filtered_train_data)
            filtered_test_data = add_q_id(filtered_test_data)
            
            # save train and test sets
            save_json(filtered_train_data, os.path.join(output_dir, f"MSD_{int(threshold*100)}_train.json"))
            save_json(filtered_test_data, os.path.join(output_dir, f"MSD_{int(threshold*100)}_test.json"))
            
            # Update the summary with the count of annotations
            if dataset not in summary:
                summary[dataset] = {}
            summary[dataset][f"threshold_{int(threshold*100)}"] = {
                "train_count": len(filtered_train_data),
                "test_count": len(filtered_test_data)
            }
    
    # Save the summary JSON
    save_json(summary, os.path.join(output_dir, "threshold_split_summary.json"))

# Run the main function
if __name__ == "__main__":
    process_datasets()
