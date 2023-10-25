from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import tensorflow as tf
import tifffile
from utils import combined_loss
from keras.models import load_model
import os

COLOR_MAP = {
    0: [255, 255, 255], # white
    1: [0, 153, 0], # dark green
    2: [198, 176, 68], # mustard
    3: [39, 255, 135], # lime green
    4: [165, 165, 165], # grey
    5: [249, 255, 164], # light yellow
    6: [28, 13, 255] # blue
    # Add more mappings as needed
}

model = load_model("./models/model_combined_loss.h5", custom_objects={'combined_loss': combined_loss})

app = FastAPI()

MAX_VAL = 4096.0
def compute_ndvi(image_array):
    red = image_array[:, :, 3]
    nir = image_array[:, :, 7]

    ndvi = (nir - red) / (nir + red + 1e-10)  

    return ndvi


def numpy_to_base64_rgb(numpy_img):
    img = Image.fromarray(numpy_img.astype('uint8'), 'RGB')
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def create_mask_rgb(label_img):
    labels = np.unique(label_img)
    label_rgb_mask = {}
    for label in labels:
        rgb_array = np.zeros((*label_img.shape, 3), dtype=np.uint8)
        mask = (label_img == label)
        rgb_array[mask] = COLOR_MAP[label]
        label_rgb_mask[int(label)] = numpy_to_base64_rgb(rgb_array)
    return label_rgb_mask

def preprocess_image(contents: bytes, desired_shape: tuple) -> np.ndarray:
    #with open("temp_check.tif", "wb") as f:
    #    f.write(contents)
   
    #with tifffile.TiffFile("temp_check.tif") as tif:
    #    img = tif.asarray()
    with tifffile.TiffFile(BytesIO(contents)) as tif:
        img = tif.asarray()
    img_shape = img.shape

    if img_shape[2] != desired_shape[2]:
        raise HTTPException(status_code=400, detail="Incorrect number of bands in the image.")
    if img_shape[0] != desired_shape[0] or img_shape[1] != desired_shape[1]:
        raise HTTPException(status_code=400, detail="Incorrect image size.")

    ndvi = compute_ndvi(img)
    ndvi = tf.expand_dims(ndvi, axis=-1)
    ndvi = (ndvi + 1.0) / 2.0
    ndvi = tf.clip_by_value(tf.cast(ndvi, tf.float32), 0., 1.)
    img = tf.clip_by_value(tf.cast(img, tf.float32) / MAX_VAL, 0., 1.)
    
    combined_img = tf.concat([img, ndvi], axis=-1)
    #os.remove("temp_check.tif")
    return combined_img

def generate_prediction(image1: np.ndarray, image2: np.ndarray):
    combined_input = np.stack([image1, image2], axis=0)
    
    predicted_labels = model.predict(combined_input)
    predicted_labels = np.argmax(predicted_labels, axis=-1)

    return predicted_labels

@app.post("/upload/", response_class=JSONResponse, tags=["image-processing"])
async def upload_files(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    contents1 = await file1.read()
    contents2 = await file2.read()

    preprocessed_image1 = preprocess_image(contents1, (256, 256, 13))
    preprocessed_image2 = preprocess_image(contents2, (256, 256, 13))

    label1, label2 = generate_prediction(preprocessed_image1, preprocessed_image2)
    

    label_mask_dict = {}
    label_mask_dict[0] = create_mask_rgb(label1)
    label_mask_dict[1] = create_mask_rgb(label2)

    response = {"status":"success"}
    response.update(label_mask_dict)

    return JSONResponse(content=response)



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
