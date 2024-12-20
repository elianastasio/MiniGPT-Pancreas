## Dataset Preparation for MiniGPT-Pancreas

### MSD_PANCREAS

**1. Download the pancreas dataset (task 7) from [Google Drive](https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2)
**2. Place imagesTr and labelsTr in the "datasets/MSD" directory
**3. Execute the follwing scripts to prepare the image-text pairs
```bash
cd datasets/MSD/scripts
python generate_slices_info.py # to extract info from 3D volumes
python save_slices.py          # to save slices as PNG images
python generate_jsons.py       # to generate JSON files for training and testing
```
