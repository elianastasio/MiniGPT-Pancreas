import json
import random
import os

#random.seed(42)

# Input file paths
msd_test_path = "../../MSD/annotations/MSD_tumor_test.json"
nih_test_path = "../../NIH/annotations/NIH_test.json"
msd_train_path = "../../MSD/annotations/MSD_tumor_train.json"
nih_train_path = "../../NIH/annotations/NIH_train.json"

# Output file paths
tc_test_path = "../annotations/TC_test.json"
tc_train_path = "../annotations/TC_train.json"

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def build_balanced_dataset(msd_data, nih_data):
    n_tumor = len(msd_data)
    n_no_tumor = len(nih_data)

    n_samples = min(n_tumor, n_no_tumor)
    nih_sampled = random.sample(nih_data, n_samples)

    merged = []
    q_id_counter = 1

    def process_entry(entry, label):
        nonlocal q_id_counter
        entry["answer"] = label
        entry["q_id"] = q_id_counter
        q_id_counter += 1
        # Compute height and width from bbox_pancreas
        bbox = entry.get("bbox_pancreas", [0, 0, 0, 0])
        entry["height"] = bbox[3] - bbox[1]
        entry["width"] = bbox[2] - bbox[0]
        merged.append(entry)

    for entry in msd_data:
        process_entry(entry, "yes")

    for entry in nih_sampled:
        process_entry(entry, "no")

    return merged

if __name__ == "__main__":
    # Load JSONs
    msd_test = load_json(msd_test_path)
    nih_test = load_json(nih_test_path)
    msd_train = load_json(msd_train_path)
    nih_train = load_json(nih_train_path)

    # Build balanced datasets
    tc_test = build_balanced_dataset(msd_test, nih_test)
    tc_train = build_balanced_dataset(msd_train, nih_train)

    # Save to files
    save_json(tc_test, tc_test_path)
    save_json(tc_train, tc_train_path)

    print(f"Test set saved to {tc_test_path}")
    print(f"Train set saved to {tc_train_path}")
