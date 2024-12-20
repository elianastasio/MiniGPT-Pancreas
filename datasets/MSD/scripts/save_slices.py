import json
import os
import numpy as np
import nibabel as nib
import pydicom
import matplotlib.pyplot as plt
from tqdm import tqdm
import re
from skimage import exposure

# preprocess the image (histogram clipping and adjustment)
def preprocess_image(slice_data):
    # Apply histogram clipping and adjustment
    p2, p98 = np.percentile(slice_data, (2, 98))
    slice_data = exposure.rescale_intensity(slice_data, in_range=(p2, p98))
    #apply histogram equalization
    slice_data = exposure.equalize_hist(slice_data)
    
    return slice_data

# read json data
with open('../annotations/slices_info.json') as f:
    msd_data = json.load(f)

# define directories
slices_output_dir = '../slices'
volumes_path = '../imagesTr'

# process slices
def process_slices(data, volumes_path, output_dir, dataset_name, debug=False):
    previous_volume_name = None
    volume_data = None
    volume_change_flag = 1
    
    for entry in tqdm(data, desc=dataset_name):
        volume_name = entry['volume_name']
        slice_index = entry['slice_index']

        # load nifti volume if it's different from the previous one
        if volume_name != previous_volume_name:
            nifti_file = os.path.join(volumes_path, volume_name)
            nifti = nib.load(nifti_file)
            volume_data = nifti.get_fdata()
            previous_volume_name = volume_name
            volume_change_flag = 1
        else:
            volume_change_flag = 0

        # extract slice
        if (debug and volume_change_flag == 1) or not debug:
            slice_data = volume_data[:, :, slice_index]

            # preprocess slice
            processed_slice = preprocess_image(slice_data)

            # remove .nii extension from volume_name
            volume_name_without_extension = os.path.splitext(os.path.splitext(volume_name)[0])[0]

            # save slice as png
            output_filename = f"{volume_name_without_extension}_slice_{slice_index}.png"
            output_path = os.path.join(output_dir, output_filename)
            plt.imsave(output_path, processed_slice, cmap='gray')

# Set debug mode
debug = False

# process MSD dataset
process_slices(msd_data, volumes_path, slices_output_dir, "MSD", debug=debug)

print("MSD Slices saved successfully.")
