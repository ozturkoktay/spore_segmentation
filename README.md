# Fern Spore Analysis

## Project Overview
This project aims to analyze and categorize fern spores using machine learning and deep learning techniques. It includes scripts for data augmentation, clustering (DBSCAN and K-Means), image segmentation, and a GUI for user-friendly interaction. The goal is to develop an image labeling tool to make the labeling faster using the previously trained model.

---

## Directory Structure

```

.
├── augment.py                 # Data augmentation script
├── classic_apprch.py          # Traditional analysis approach
├── dataset                    # Folder containing all data files and organized datasets
│   ├── 10k_split              # Split data samples
│   ├── augmented_spores       # Augmented spore images
│   ├── organized              # Structured datasets for easy access
│   └── other folders          # Additional data files and categories
├── dbscan.py                  # DBSCAN clustering implementation
├── FernSpores.csv             # Main CSV dataset of spore information
├── GUI                        # GUI application files
│   ├── app.py                 # Main script to run the GUI
│   ├── model.py               # Core model for spore classification
│   ├── templates              # HTML templates for GUI
│   └── static                 # Static files for the GUI (e.g., CSS, JS)
├── kmeans.py                  # K-Means clustering script
├── opencv_sa.py               # OpenCV-based image segmentation script
├── resnet_train.py            # ResNet model training script
├── train_resnet.ipynb         # Jupyter notebook for training ResNet interactively
└── utility.py                 # Helper functions

```
---

## Setup and Installation

### Prerequisites
Ensure that you have **Python 3.x** installed and install required libraries with:

`pip install -r requirements.txt`

### Installation Steps
1. Clone the Repository:

    `git clone https://github.com/your-repo/fern-spore-analysis.git cd fern-spore-analysis`

2. Install Dependencies:

    `pip install -r requirements.txt`

3. Data Preparation:
   
Unzip any necessary dataset files, if required:

`unzip dataset/10k_split.zip -d dataset/10k_split`

---

## Usage Guide

### Running Scripts

**Data Augmentation**: Run `augment.py` to perform data augmentation on fern spore images.

`python augment.py`

**Clustering**:
Use `dbscan.py` or `kmeans.py` to perform clustering analysis on the dataset.

**Image Segmentation**:
Run `opencv_sa.py` for segmenting fern spore images with image processing.

**ResNet Model Training**:
Train the ResNet-based model using either the script or the interactive notebook (`python resnet_train.py`, `train_resnet.ipynb`).

### GUI Usage

The project includes a GUI for a more interactive analysis experience.

1. Navigate to the GUI Folder:

    `cd GUI`

2. Launch the GUI Application:
Run the GUI with the following command:

    `python app.py`

3. Using the GUI:
The GUI allows users to label their spores either `unknown`, `monolete`, or `trilete`. At the beginning, trained spore ResNet model will make the initial predictions then user will only see low confidence prediction which are labeled as `unknown`. The user will label these spores by clicking on the image and choosing a label from the provided list. 

---

## Code Structure and Explanation

- **augment.py**: Script for performing data augmentation on images, generating new samples for training.
- **classic_apprch.py**: Contains a traditional (non-deep learning) approach for fern spore analysis.
- **dataset/**: Folder for all datasets, organized as needed for the project.
   - **10k_split/**: Sample split data.
   - **augmented_spores/**: Augmented spore images.
   - **organized/**: Well-organized dataset for model training.
- **dbscan.py** & **kmeans.py**: Scripts implementing DBSCAN and K-Means clustering for categorizing spores.
- **FernSpores.csv**: Main dataset of spore features and information.
- **GUI/app.py**: The main file to run the GUI. Launches the user interface for interactive analysis and model prediction on spore images.
- **GUI/model.py** and **GUI/model_cpy.py**: Model scripts for spore classification, using trained weights to perform predictions.
- **GUI/model_weights/**: Folder containing pretrained model files used by **model.py**.
- **GUI/static/**: Contains static assets for the GUI.
  - **images/**: Categorized folders for spore images (monolete, trilete, unknown).
  - **styles.css**: Styling for the GUI.
- **GUI/templates/**: HTML templates for the GUI interface.
  - **index.html**: Main HTML template for the GUI layout.
- **GUI/analysis/**: Collection of analysis plots and images for visual insights into confidence scores and distribution.
- **GUI/segment.ipynb**: Jupyter notebook for exploring segmentation techniques on spore images.
- **opencv_sa.py**: Image segmentation using OpenCV functions for preprocessing.
- **resnet_train.py**: Script to train a ResNet model for spore classification.
- **train_resnet.ipynb**: Jupyter notebook for interactive ResNet model training.
- **utility.py**: Utility functions used across scripts.
