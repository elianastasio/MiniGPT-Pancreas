## Dataset Preparation for MiniGPT-Pancreas

### MSD_PANCREAS

**1.** Download the MSD pancreas dataset from [Google Drive] (task 7) (https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2)  
**3.** Extract the downloaded file  
**2.** Place imagesTr and labelsTr in "datasets/MSD/raw_dataset"  
**3.** Execute the follwing scripts in order to prepare the image-text pairs
```bash
cd datasets/MSD/scripts
python MSD_generate_slices_info.py # to extract info from 3D volumes
python MDS_generate_jsons.py       # to generate JSON files for training and testing
python MSD_save_slices.py          # to save slices as PNG images
```
