import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

from analysis import quantify_changes
from typing import List
from report import zip_folder, save_overlayed_images
from s2cloudless import S2PixelCloudDetector
from utils import custom_loss, normalize, numpy_to_base64_rgb, loading_tiff_and_resize, model_dir_name

COLOR_MAP = {
    0: [255, 255, 255],  # white
    1: [0, 153, 0],      # dark green
    2: [198, 176, 68],   # mustard
    3: [39, 255, 135],   # lime green
    4: [165, 165, 165],  # grey
    5: [249, 255, 164],  # light yellow
    6: [28, 13, 255]     # blue
}
MAX_VAL = 4096.0

model = load_model(
    f"{model_dir_name}/model_custom_loss_bs_8_ep_100.h5",
    custom_objects={'loss': custom_loss}
)
cloud_detector = S2PixelCloudDetector(
    threshold=0.996, average_over=4, dilation_size=2, all_bands=True
)


def compute_ndvi(image_array):
    """Computes the Normalized Difference Vegetation Index (NDVI) for the given image array."""
    red = image_array[:, :, 3]
    nir = image_array[:, :, 7]

    ndvi = (nir - red) / (nir + red + 1e-10)

    return ndvi


def preprocess_image(img: np.ndarray) -> np.ndarray:
    """Preprocesses the image for prediction."""
    ndvi = compute_ndvi(img)
    ndvi = tf.expand_dims(ndvi, axis=-1)
    ndvi = (ndvi + 1.0) / 2.0
    ndvi = tf.clip_by_value(tf.cast(ndvi, tf.float32), 0., 1.)
    #added to handle channels more than 13
    if img.shape[2] >= 14:
        img = img[:,:,:-1]
    
    img = tf.clip_by_value(tf.cast(img, tf.float32) / MAX_VAL, 0., 1.)

    combined_img = tf.concat([img, ndvi], axis=-1)

    return combined_img


def generate_prediction(image1: np.ndarray, image2: np.ndarray):
    """Generates a prediction for the given image."""
    combined_input = np.stack([image1, image2], axis=0)

    predicted_labels = model.predict(combined_input)
    predicted_labels = np.argmax(predicted_labels, axis=-1)

    return predicted_labels


def create_mask_rgb(label_img):
    """Creates an RGB mask for each label in the label image."""
    labels = np.unique(label_img)
    print(f"Unique labels: {labels}")
    label_masks = {}
    for label in labels:
        mask = (label_img == label)
        label_masks[label] = mask

    return label_masks


def image_overlay_with_mask(image: np.ndarray, label: int, mask: np.ndarray, alpha: float = 0.5):
    """Combines image and its segmentation mask into a single image."""
    # Get label specific color array
    color = np.asarray(COLOR_MAP[label])
    # Create an RGB version of the mask using the specified color
    colored_mask = np.zeros_like(image)
    for i in range(3):
        colored_mask[:, :, i] = mask * color[i]
    # Combine the mask and the image
    image_combined = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)

    return image_combined


def get_masked_images(image: np.ndarray, masks: dict):
    """Combines image and its segmentation mask into a single image."""
    ret = {
        label: image_overlay_with_mask(image, label, mask)
        for label, mask in masks.items()
    }

    return ret


def encode_images_with_base64(masked_images: dict, original_image: np.ndarray):
    """Encodes the images with base64."""
    # encoded_images = [numpy_to_base64_rgb(img) for img in images]
    encoded_images = []
    for i in range(7):
        img = masked_images.get(i)
        encoded = numpy_to_base64_rgb(img) if img is not None else ""
        encoded_images.append(encoded)

    encoded_images.append(numpy_to_base64_rgb(original_image))

    return encoded_images


def normalize_and_get_cloud_masks(image: np.ndarray):
    """Returns the cloud masks for the given image."""
    # Normalize image
    normalized_image = normalize(image)
    # Get cloud masks
    cloud_masks = cloud_detector.get_cloud_masks(normalized_image[np.newaxis, ...])

    return cloud_masks


async def deforestation_detection(images: List[bytes], filenames: List[str]):
    """Performs deforestation detection on the given images."""
    # Load TIFF images
    image_1 = loading_tiff_and_resize(images[0], (256, 256, 13))
    image_2 = loading_tiff_and_resize(images[1], (256, 256, 13))
    # Cloud detection
    cloud_mask_1 = normalize_and_get_cloud_masks(image_1)
    cloud_mask_2 = normalize_and_get_cloud_masks(image_2)
    # Preprocess for model prediction
    preprocessed_image_1 = preprocess_image(image_1)
    preprocessed_image_2 = preprocess_image(image_2)
    # Model prediction
    label_1, label_2 = generate_prediction(preprocessed_image_1, preprocessed_image_2)
    label_1_declouds = label_1 * np.logical_not(cloud_mask_1[0])
    label_2_declouds = label_2 * np.logical_not(cloud_mask_2[0])
    # Create masks
    masks_1, masks_2 = create_mask_rgb(label_1), create_mask_rgb(label_2)
    masks_1_declouds, masks_2_declouds = create_mask_rgb(label_1_declouds), create_mask_rgb(label_2_declouds)
    full_masks_1 = {**masks_1_declouds, **masks_1}
    full_masks_2 = {**masks_2_declouds, **masks_2}
    # Get masked images
    # Pick RGB channels of the original image, and map to 0-255
    rgb_image_1 = np.dstack([normalize(image_1[:, :, i]) * 255 for i in [3, 2, 1]])
    rgb_image_2 = np.dstack([normalize(image_2[:, :, i]) * 255 for i in [3, 2, 1]])
    masked_images_1 = get_masked_images(rgb_image_1, full_masks_1)
    masked_images_2 = get_masked_images(rgb_image_2, full_masks_2)
    # Convert to base64
    masked_images_1_enc = encode_images_with_base64(masked_images_1, rgb_image_1)
    masked_images_2_enc = encode_images_with_base64(masked_images_2, rgb_image_2)
    # Save overlayed images
    save_overlayed_images(masked_images_1, filenames[0])
    save_overlayed_images(masked_images_2, filenames[1])
    # Quantify deforestation
    changes = quantify_changes(label_1_declouds, label_2_declouds)
    # Zip all files
    zip_folder("analysis")

    # Return
    results = {
        "first": {
            "default": 7,
            "images": masked_images_1_enc
        },
        "second": {
            "default": 7,
            "images": masked_images_2_enc
        },
        "changes": changes
    }

    return results
