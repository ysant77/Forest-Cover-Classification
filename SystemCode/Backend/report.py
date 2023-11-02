import os
from zipfile import ZipFile

import cv2

LABEL_MAP = [
    "Cloud", "Forest", "Grassland",
    "Wetland", "Urban", "Barren", "Water"
]


def get_all_file_paths(directory):
    file_paths = []

    for root, directories, files in os.walk(directory):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)

    return file_paths


def zip_folder(directory: str):
    # calling function to get all file paths in the directory
    file_paths = get_all_file_paths(directory)

    # printing the list of all files to be zipped
    print('Following files will be zipped:')
    for file_name in file_paths:
        print(file_name)

        # writing files to a zipfile
    with ZipFile('files/analysis.zip', 'w') as zip_obj:
        # writing each file one by one
        for file in file_paths:
            zip_obj.write(file)

    print('All files zipped successfully!')


def save_overlayed_images(images: dict, input_image_name: str):
    """Saves the overlayed images to the disk."""
    path = f"analysis/{input_image_name}"
    labels = LABEL_MAP.copy()
    # Create directory if not exists
    if not os.path.exists(path):
        os.makedirs(path)
    # Save images
    print(f"Labels: {images.keys()}")
    for label, image in images.items():
        save_path = f"{path}/{labels[label]}.png"
        print(f"Saving picture to: {save_path}")
        cv2.imwrite(save_path, image)
    # Done
    print(f"Done saving processed images for: {input_image_name}")
