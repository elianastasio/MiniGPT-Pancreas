## Dataset Preparation for MiniGPT-Pancreas

To preprocess the datasets used in this project, you can either run all scripts individually or run the full pipeline with a single command:

```bash
cd datasets
python dataset_runner.py
```

**Important**: You still need to download and place the raw datasets in their respective folders according to the instructions in the first steps of each dataset section below. The runner script assumes the raw data is already in place.

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
python MSD_tumor_generate_jsons.py # for tumor detection
python MSD_tumor_save_slices.py
```
**Note:** You can add the --balanced flag to the generate_jsons and save_slices scripts to add non-pancreas or non-tumor slices. Used for balanced training.

### NIH_PANCREAS

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
**Note:** You can add the --balanced flag to the generate_jsons and save_slices scripts to add non-pancreas slices. Used for balanced training.

### Tumor Classification

**1.** Execute the follwing scripts in order to prepare the image-text pairs
```bash
cd datasets/TC/scripts          
python TC_generate_jsons.py       # to generate JSON files for training and testing
python TC_save_slices.py          # to save slices as PNG images
```

### AbdomenCT-1K

**1.** Download the AbdomenCT-1K dataset by following the instructions [here](https://github.com/JunMa11/AbdomenCT-1K?tab=readme-ov-file). Place the 3 parts AbdomenCT-1K-ImagePart*.zip and the mask file (donwloadable from the same page as the third part) in datasets/ABD/raw_dataset.
**2.** Extract the four files
**3.** Execute the follwing scripts in order to prepare the image-text pairs
```bash
cd datasets/ABD/scripts
python ABD_generate_slices_info.py # to extract info from 3D volumes
python ABD_generate_jsons.py       # to generate JSON files for training and testing
python ABD_save_slices.py          # to save slices as PNG images
```