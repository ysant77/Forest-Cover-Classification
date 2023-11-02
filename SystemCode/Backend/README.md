# 🌲 Deforestation Detection

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
* `analysis` - Analysis results folder.
* `Sample_Images` - Sample images folder.

```
Deforestation_Detection_Backend
├── README.md
├── Sample_Images
│   ├── S2L1C_Kranji_20200106.tif
│   ├── S2L1C_Kranji_20231007.tif
│   ├── SCL_Kranji_20200106.tif
│   └── SCL_Kranji_20231007.tif
├── analysis
│   ├── S2L1C_Kranji_20200106
│   └── S2L1C_Kranji_20231007
├── analysis.py
├── files
│   ├── analysis.zip
│   └── changes.csv
├── main.py
├── models
│   ├── model_combined_loss.h5
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

1. First, follow the models/model_file.txt to download the pre-trained models.
2. Install the required packages.
3. Run the server application with the following command.

```bash
uvicorn main:app --reload --host 0.0.0.0
```
