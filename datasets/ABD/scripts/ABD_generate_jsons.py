import os
import json
import random
import argparse
from tqdm import tqdm

labels_map = {
    1: "liver",
    2: "kidney",
    3: "spleen",
    4: "pancreas"
}

def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def save_json(data, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def split_volumes(volume_names, test_ratio=0.2):
    #random.seed(42)
    shuffled = volume_names[:]
    random.shuffle(shuffled)
    test_size = int(len(shuffled) * test_ratio)
    return shuffled[test_size:], shuffled[:test_size]

def add_q_id(data):
    for i, item in enumerate(data, start=1):
        item["q_id"] = i
    return data

def generate_entries(volumes_subset, volumes_dict, threshold=0.6):
    entries = []
    for vol in tqdm(volumes_subset, desc="Processing volumes"):
        for slice_item in volumes_dict[vol]:
            for label, organ in labels_map.items():
                if slice_item[f"{organ}_bbox_ratio"] > threshold:
                    entry = {
                        "dataset": slice_item["dataset"],
                        "volume_name": slice_item["volume_name"],
                        "slice_id": slice_item["slice_id"],
                        "slice_index": slice_item["slice_index"],
                        "slice_count": slice_item["slice_count"],
                        "width": slice_item["width"],
                        "height": slice_item["height"],
                        "organ": organ,
                        "pixels": slice_item[f"pixels_{organ}"],
                        "pixels_ratio": slice_item[f"{organ}_pixels_ratio"],
                        "bbox": slice_item[f"bbox_{organ}"],
                        "bbox_ratio": slice_item[f"{organ}_bbox_ratio"]
                    }
                    entries.append(entry)
    return entries

def balance_entries_overall(entries):
    # Group by organ
    organ_groups = {organ: [] for organ in labels_map.values()}
    for e in entries:
        organ_groups[e["organ"]].append(e)

    # Find minimum count across organs
    min_count = min(len(lst) for lst in organ_groups.values())

    # Sample each organ to have min_count entries
    balanced_entries = []
    for organ, lst in organ_groups.items():
        if len(lst) > min_count:
            balanced_entries.extend(random.sample(lst, min_count))
        else:
            balanced_entries.extend(lst)
    return balanced_entries

def process_abd_splits_balanced(input_json, threshold=0.6):
    #random.seed(42)
    data = load_json(input_json)

    # Only consider volumes present in Masks folder
    mask_dir = "../raw_dataset/Masks"
    available_volumes = set(f.replace(".nii.gz", "") for f in os.listdir(mask_dir) if f.endswith(".nii.gz"))
    data = [s for s in data if s["volume_name"].replace("_0000.nii.gz", "") in available_volumes]

    # Group slices by volume
    volumes_dict = {}
    for item in data:
        vol_name = item["volume_name"]
        volumes_dict.setdefault(vol_name, []).append(item)

    volume_names = list(volumes_dict.keys())
    train_volumes, test_volumes = split_volumes(volume_names, test_ratio=0.2)

    print("Generating train entries...")
    train_entries = generate_entries(train_volumes, volumes_dict, threshold)
    print("Generating test entries...")
    test_entries = generate_entries(test_volumes, volumes_dict, threshold)

    print("Balancing train entries across organs...")
    train_entries = balance_entries_overall(train_entries)
    print("Balancing test entries across organs...")
    test_entries = balance_entries_overall(test_entries)

    train_entries = add_q_id(train_entries)
    test_entries = add_q_id(test_entries)

    output_dir = "../annotations"
    save_json(train_entries, os.path.join(output_dir, "ABD_train.json"))
    save_json(test_entries, os.path.join(output_dir, "ABD_test.json"))

    # Report counts per organ
    def count_organs(entries):
        counts = {organ: 0 for organ in labels_map.values()}
        for e in entries:
            counts[e["organ"]] += 1
        return counts

    print("Train counts per organ:", count_organs(train_entries))
    print("Test counts per organ:", count_organs(test_entries))
    print(f"Train JSON saved: {len(train_entries)} entries")
    print(f"Test JSON saved: {len(test_entries)} entries")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, default="../annotations/ABD_slices_info.json")
    parser.add_argument("--threshold", type=float, default=0.6)
    args = parser.parse_args()
    process_abd_splits_balanced(args.input_json, threshold=args.threshold)
