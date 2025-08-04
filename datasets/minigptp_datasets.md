## Dataset Preparation for MiniGPT-Pancreas

### MSD_PANCREAS

**1.** Download the MSD pancreas dataset from [Google Drive](https://drive.google.com/drive/folders/1HqEgzS8BV2c7xYNrZdEAnrHk7osJJ--2) (task 7)   
**2.** Extract the downloaded file  
**3.** Place imagesTr and labelsTr in "datasets/MSD/raw_dataset"  
**4.** Execute the follwing scripts in order to prepare the image-text pairs
```bash
cd datasets/MSD/scripts
python MSD_generate_slices_info.py # to extract info from 3D volumes
python MSD_generate_jsons.py       # to generate JSON files for training and testing
python MSD_save_slices.py          # to save slices as PNG images
```

### MSD_PANCREAS

**1.** Download the NIH TCIA pancreas dataset from [here](https://www.cancerimagingarchive.net/collection/pancreas-ct/). You will need to use [NBIA Data Retriever](https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images)   
**2.** Place the 80 "PANCREAS_00**" folders in "datasets/NIH/raw_dataset/volumes"  
**3.** Place the labels folder "TCIA_pancreas_labels-02-05-2017" in "dataset/NIH/raw_dataset"  
**4.** Execute the follwing scripts in order to prepare the image-text pairs
```bash
cd datasets/NIH/scripts
python NIH_generate_slices_info.py # to extract info from 3D volumes
python NIH_generate_jsons.py       # to generate JSON files for training and testing
python NIH_save_slices.py          # to save slices as PNG images
```