import os
import json
from PIL import Image
from tqdm import tqdm

# Paths to the new merged JSONs
tc_train_json_path = "../annotations/TC_train.json"
tc_test_json_path = "../annotations/TC_test.json"

# Output folder for cropped slices
output_dir = "../slices"
os.makedirs(output_dir, exist_ok=True)

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def get_image_path(entry):
    """
    Constructs the image path based on the dataset label in the entry.
    """
    dataset = entry.get("dataset", None)
    volume_name = entry["volume_name"]
    slice_idx = entry["slice_index"]

    if dataset == "MSD":
        img_name = f"{volume_name.replace('.nii.gz', '')}_slice_{slice_idx}.png"
    elif dataset == "NIH":
        vol_idx = volume_name.replace("label", "").replace(".nii.gz", "")
        img_name = f"PANCREAS_{vol_idx}_slice_{slice_idx}.png"
    else:
        raise ValueError(f"Unknown dataset for entry: {dataset}")

    return os.path.join("../../", dataset, "slices", img_name)

def crop_and_save_image(image_path, bbox, output_path):
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    img = Image.open(image_path)
    cropped = img.crop(bbox)
    cropped.save(output_path)

def process_json(json_path, dataset_name, threshold=0.6):
    json_data = load_json(json_path)
    for entry in tqdm(json_data, desc=f"Cropping {dataset_name}"):
        bbox = entry["bbox_pancreas"]
        image_path = get_image_path(entry)
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        crop_and_save_image(image_path, bbox, output_path)

if __name__ == "__main__":
    process_json(tc_train_json_path, "TC_train")
    process_json(tc_test_json_path, "TC_test")

    print(f"TC slices saved successfully in {output_dir}")
