# 🌲 Forest Cover Classification

Backend system of PRS Practice Module, Group 21, NUS-ISS, 2023.

## Project Structure

* `main.py` - Main entry point of the server application.
* `segmentation.py` - Land cover preprocessing and classification module. This is the core module of the backend system.
* `s2cloudless` - Cloud detection module.
* `analysis.py` - Time series analysis module.
* `report.py` - Report generation module.
* `utils.py` - Utility functions.
* `models` - Pre-trained models folder.
* `files` - Temporary file folder.
* `analysis` - Analysis results folder. (created automatically by code)
* `Sample_Images` - Sample images folder. (created automatically by code)

```
Deforestation_Detection_Backend
├── README.md
├── analysis.py
├── files
│   ├── analysis.zip
│   └── changes.csv
├── main.py
├── models
│   ├── model_custom_loss_bs_8_ep_100.h5
│   ├── model_file.txt
│   └── pixel_s2_cloud_detector_lightGBM_v0.1.txt
├── report.py
├── s2cloudless
│   ├── __init__.py
│   ├── cloud_detector.py
│   ├── pixel_classifier.py
│   └── utils.py
├── segmentation.py
└── utils.py
```

## Getting Started

1. First, follow the models/model_file.txt to download the pre-trained models. After downloading, keep the model files in models directory
2. Install the required packages using the command ```pip install -r requirements.txt```. 
3. Run the server application with the following command from this directory.

## Note:
1. This system has been implemented and tested using python 3.8, and hence recommended version of python is 3.8
2. If any issues are encountered while installing requirements, run the command ```pip install -r req_win.txt```

```bash
uvicorn main:app --reload --host 0.0.0.0
```
