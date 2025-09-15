import os
import json
import nibabel as nib
import numpy as np
from tqdm import tqdm
from scipy.ndimage import label as nd_label

labels_map = {
    1: "liver",
    2: "kidney",
    3: "spleen",
    4: "pancreas"
}

def get_bbox(segmentation_array, label):
    indices = np.where(segmentation_array == label)
    if len(indices[0]) == 0:
        return None
    x_min = np.min(indices[1])
    y_min = np.min(indices[0])
    x_max = np.max(indices[1])
    y_max = np.max(indices[0])
    return [x_min, y_min, x_max, y_max]

def get_bbox_area(bbox):
    if bbox is None:
        return 0
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return width * height

def get_right_kidney_bbox(slice_array):
    """Return bounding box for right kidney only."""
    kidney_mask = (slice_array == 2).astype(np.uint8)
    if np.sum(kidney_mask) == 0:
        return None

    # Label connected components
    labeled_array, num_features = nd_label(kidney_mask)
    component_areas = []
    component_coords = []

    for i in range(1, num_features + 1):
        coords = np.where(labeled_array == i)
        area = len(coords[0])
        component_areas.append(area)
        component_coords.append(coords)

    if num_features == 1:
        # Only one kidney, check left side of image
        x_coords = component_coords[0][1]
        if np.mean(x_coords) < slice_array.shape[1] // 2:  # left side
            return get_bbox_for_coords(component_coords[0])
        else:
            return None  # it's not the right kidney
    elif num_features == 2:
        # Two components, choose the left one
        left_index = np.argmin([np.min(c[1]) for c in component_coords])
        return get_bbox_for_coords(component_coords[left_index])
    else:
        # More than two components: pick 2 largest and choose left one
        sorted_indices = np.argsort(component_areas)[-2:]
        left_index = sorted_indices[np.argmin([np.min(component_coords[i][1]) for i in sorted_indices])]
        return get_bbox_for_coords(component_coords[left_index])

def get_bbox_for_coords(coords):
    x_min = np.min(coords[1])
    y_min = np.min(coords[0])
    x_max = np.max(coords[1])
    y_max = np.max(coords[0])
    return [x_min, y_min, x_max, y_max]

def save_slices_data_abdomen():
    slices_info = []
    slice_id = 1

    data_dirs = [
        "../raw_dataset/AbdomenCT-1K-ImagePart1",
        "../raw_dataset/AbdomenCT-1K-ImagePart2",
        "../raw_dataset/AbdomenCT-1K-ImagePart3"
    ]
    mask_dir = "../raw_dataset/Masks"

    volume_files = []
    for d in data_dirs:
        files = [os.path.join(d, f) for f in os.listdir(d) if f.endswith(".nii.gz") and not f.startswith(".")]
        volume_files.extend(sorted(files))

    for volume_path in tqdm(volume_files, desc="Processing volumes"):
        volume_name = os.path.basename(volume_path)
        case_id = volume_name.split("_")[1]
        mask_name = f"Case_{case_id}.nii.gz"
        mask_path = os.path.join(mask_dir, mask_name)

        if not os.path.exists(mask_path):
            #print(f"Warning: Mask for {volume_name} not found. Skipping.")
            continue

        mask_image = nib.load(mask_path)
        mask_array = mask_image.get_fdata()
        mask_array = np.rot90(mask_array, k=1, axes=(0, 1))
        mask_array = np.flip(mask_array, axis=1)

        slices_count = mask_array.shape[2]

        organ_pixel_counts = {label: [] for label in labels_map}
        organ_bbox_areas = {label: [] for label in labels_map}

        for slice_array in np.rollaxis(mask_array, 2):
            for label in labels_map:
                if label == 2:  # kidney
                    bbox = get_right_kidney_bbox(slice_array)
                    pixel_count = np.sum(slice_array == 2 if bbox is not None else 0)
                else:
                    pixel_count = np.sum(slice_array == label)
                    bbox = get_bbox(slice_array, label)

                organ_pixel_counts[label].append(pixel_count)
                organ_bbox_areas[label].append(get_bbox_area(bbox))

        max_pixels = {label: max(counts) if counts else 1 for label, counts in organ_pixel_counts.items()}
        max_bboxes = {label: max(areas) if areas else 1 for label, areas in organ_bbox_areas.items()}

        for idx, slice_array in enumerate(np.rollaxis(mask_array, 2)):
            slice_info = {
                "dataset": "ABD",
                "volume_name": volume_name,
                "slice_id": int(slice_id),
                "slice_index": int(idx),
                "slice_count": int(slices_count),
                "width": int(slice_array.shape[1]),
                "height": int(slice_array.shape[0])
            }

            for label, organ_name in labels_map.items():
                if label == 2:  # kidney
                    bbox = get_right_kidney_bbox(slice_array)
                    pixel_count = np.sum(slice_array == 2 if bbox is not None else 0)
                else:
                    pixel_count = organ_pixel_counts[label][idx]
                    bbox = get_bbox(slice_array, label)

                bbox_coords = [int(c) for c in bbox] if bbox else [0, 0, 0, 0]
                slice_info[f"pixels_{organ_name}"] = int(pixel_count)
                slice_info[f"bbox_{organ_name}"] = bbox_coords
                slice_info[f"{organ_name}_pixels_ratio"] = round(pixel_count / max_pixels[label], 2) if max_pixels[label] != 0 else 0
                slice_info[f"{organ_name}_bbox_ratio"] = round(get_bbox_area(bbox) / max_bboxes[label], 2) if max_bboxes[label] != 0 else 0
                
            slices_info.append(slice_info)
            slice_id += 1

    output_path = "../annotations/ABD_slices_info.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(slices_info, f, indent=4)

    print("Data collection complete.")

if __name__ == "__main__":
    save_slices_data_abdomen()
