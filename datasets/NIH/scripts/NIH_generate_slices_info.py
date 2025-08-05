import os
import json
import nibabel as nib
import numpy as np
from PIL import Image
from tqdm import tqdm

def get_bbox(segmentation_array, label):
    if isinstance(label, tuple):
        combined_mask = np.zeros_like(segmentation_array, dtype=bool)
        for lbl in label:
            combined_mask = np.logical_or(combined_mask, segmentation_array == lbl)
        indices = np.where(combined_mask)
    else:
        indices = np.where(segmentation_array == label)
    
    if len(indices[0]) == 0:
        return None
    
    x_min = np.min(indices[1])
    y_min = np.min(indices[0])
    x_max = np.max(indices[1])
    y_max = np.max(indices[0])
    
    return [x_min, y_min, x_max, y_max]

def get_bbox_area(bbox):
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    area = width * height
    return area

def save_slices_data(data_dir):
    slices_info = []
    slice_id = 1
    volume_files = sorted(
        f for f in os.listdir(data_dir) 
        if f.endswith('.nii.gz') and not f.startswith('.')
    )
    for volume_name in tqdm(volume_files, desc='Processing volumes'):
        seg_file_path = os.path.join(data_dir, volume_name)
        seg_image = nib.load(seg_file_path)
        seg_array = seg_image.get_fdata()
        seg_array = np.rot90(seg_array, k=1, axes=(0, 1)) # Compute bbox from RAS orientation
        #seg_array = np.flip(seg_array, axis=1)

        slices_count = seg_array.shape[2]  # Number of slices in the volume
        pancreas_pixel_counts = []
        bbox_pancreas_areas = []
        
        # Compute pixel counts and bounding box areas for each slice
        for slice_array in np.rollaxis(seg_array, 2):
            pancreas_pixel_count = np.sum((slice_array == 1) | (slice_array == 2))
            bbox_pancreas = get_bbox(slice_array, (1, 2))
            bbox_pancreas_area = get_bbox_area(bbox_pancreas) if bbox_pancreas else 0
            
            pancreas_pixel_counts.append(pancreas_pixel_count)
            bbox_pancreas_areas.append(bbox_pancreas_area)
        
        max_pancreas_pixel_count = max(pancreas_pixel_counts)
        max_bbox_pancreas_area = max(bbox_pancreas_areas)

        # Iterate over slices and create json info
        for idx, slice_array in enumerate(np.rollaxis(seg_array, 2)):
            total_pancreas_pixel_count = pancreas_pixel_counts[idx]
            #if total_pancreas_pixel_count > 0:
            bbox_pancreas_area = bbox_pancreas_areas[idx]

            bbox_pancreas = get_bbox(slice_array, (1, 2))
            bbox_pancreas = [int(coord) for coord in bbox_pancreas] if bbox_pancreas else [0, 0, 0, 0]
        
            pancreas_pixels_ratio = round(total_pancreas_pixel_count / max_pancreas_pixel_count, 2)

            pancreas_bbox_ratio = round(bbox_pancreas_area / max_bbox_pancreas_area, 2)

            # Save slice info
            image_info = {
                "dataset": "NIH",
                "volume_name": volume_name,
                "slice_id": int(slice_id),
                "slice_index": int(idx),
                "slice_count": int(slices_count),
                "pixels_pancreas": int(total_pancreas_pixel_count),
                "pancreas_pixels_ratio": pancreas_pixels_ratio,
                "max_pixels_pancreas": int(max_pancreas_pixel_count),
                "bbox_pancreas": bbox_pancreas,
                "pancreas_bbox_ratio": pancreas_bbox_ratio,
                "max_bbox_pancreas": int(max_bbox_pancreas_area),
                "width": int(slice_array.shape[1]),
                "height": int(slice_array.shape[0]),
            }
            slices_info.append(image_info)
            slice_id += 1

    # write slices_info to json
    output_path = '../annotations/NIH_slices_info.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as json_file:
        json.dump(slices_info, json_file, indent=4)

    print("Data collection complete")

data_directory = '../raw_dataset/TCIA_pancreas_labels-02-05-2017'

save_slices_data(data_directory)


