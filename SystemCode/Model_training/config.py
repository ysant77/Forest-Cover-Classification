# Native Libraries
import os
import json
from pathlib import Path

# Third-party Libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model, backend as K
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, Callback

import tifffile
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt


############################# CONFIG #############################
# ds_dir = os.getcwd()


HEIGHT = 256
WIDTH = 256
CHANNEL = 14 # 3, 4, 14
INPUT_SHAPE = (WIDTH, HEIGHT, CHANNEL)
# INPUT_SHAPE = (256, 256, 14)
NUM_CLASSES = 7
MAX_VAL = 4096.0

# Augmentation Config
ROTATION_FACTOR = (-0.5, 0.5)
RAND_SEED = 42

EPSILON = 1e-7
SAMPLE_WEIGHT = [EPSILON, 1.1033, 1.0629, 1.1714, 1.1273, 1.2035, 1.0667]
weights = np.array(SAMPLE_WEIGHT)

# Training Config
BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 1e-4
AUTOTUNE = tf.data.AUTOTUNE

img_aug = Sequential([
        # layers.Resizing(height=self.img_size, width=self.img_size, interpolation='bicubic'),
        layers.experimental.preprocessing.RandomContrast(factor=(0.07, 0.1), seed=RAND_SEED),
        layers.experimental.preprocessing.RandomFlip(mode='horizontal', seed=RAND_SEED),
        layers.experimental.preprocessing.RandomFlip(mode='vertical', seed=RAND_SEED),
        layers.experimental.preprocessing.RandomRotation(ROTATION_FACTOR, fill_mode='reflect', interpolation='bilinear', seed=RAND_SEED),
        layers.experimental.preprocessing.RandomTranslation(height_factor=0.15, width_factor=0.15, fill_mode='reflect', interpolation='bilinear', seed=RAND_SEED)
    ])

label_aug = Sequential([
        # layers.Resizing(height=self.img_size, width=self.img_size, interpolation='nearest'),
        layers.experimental.preprocessing.RandomFlip(mode='horizontal', seed=RAND_SEED),
        layers.experimental.preprocessing.RandomFlip(mode='vertical', seed=RAND_SEED),
        layers.experimental.preprocessing.RandomRotation(ROTATION_FACTOR, fill_mode='reflect', interpolation='nearest', seed=RAND_SEED),
        layers.experimental.preprocessing.RandomTranslation(height_factor=0.15, width_factor=0.15, fill_mode='reflect', interpolation='nearest', seed=RAND_SEED)
    ])


def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy

    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    """
    weights = tf.constant(weights, dtype=tf.float32)
    
    def loss(y_true, y_pred):
        #y_true = tf.cast(y_true, tf.float32)

        y_true = tf.one_hot(tf.squeeze(y_true, axis=-1), depth=7)
        class_loglosses = K.mean(K.categorical_crossentropy(y_true, y_pred, from_logits=False), axis=[0, 1, 2])
        return K.sum(class_loglosses * K.constant(weights))


    return loss

############################# PREPROCESSING #############################


def train_eval_dataset_gen(ds_dir):
    # ds_dir = Path(r'D:\python_workspace\NUS_ISS\PR_Project\Dataset\DFC_Public_Dataset')

    train_eval_img_dir = 'ROIs0000_*\s2_*'
    train_eval_label_dir = train_eval_img_dir.replace('s2','dfc')
    train_eval_img_dir = [i.as_posix() for i in ds_dir.rglob(f'{train_eval_img_dir}\*.tif')]
    train_eval_label_dir = [i.as_posix() for i in ds_dir.rglob(f'{train_eval_label_dir}\*.tif')]

    X_train, X_test, y_train, y_test = train_test_split(train_eval_img_dir,train_eval_label_dir,test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test


def calculate_pixel_frequencies(dataset):
    # Initialize an array to store pixel frequencies for each class
    class_counts = np.zeros(7)  # Assuming classes are labeled from 0 to 6
    
    # Iterate over the dataset to count pixel frequencies
    for _, label in dataset:  # Assuming dataset yields (image, label) pairs
        unique, counts = np.unique(label, return_counts=True)
        for u, c in zip(unique, counts):
            class_counts[int(u)] += c

    return class_counts
def compute_class_weights(pixel_frequencies):
    # Compute class weights using the 'balanced' approach
    total_pixels = np.sum(pixel_frequencies)
    class_weights = total_pixels / (7 * pixel_frequencies)  # 7 classes

    return class_weights


def pre_process_label(label):

    label[(label==4) | (label==6)] = 2 # shrubland, savanna, grassland, croplands
    label[label==5] = 3 # wetlands
    label[label==7] = 4 # urban
    label[label==9] = 5 # urban
    label[label==10] = 6
    return label


def compute_ndvi(image_array):
    # Assuming Red band is the first channel (0th index) and NIR is the second channel (1st index)
    red = image_array[:, :, 3]
    nir = image_array[:, :, 7]

    # Compute NDVI
    ndvi = (nir - red) / (nir + red + 1e-10)  # Small value added to the denominator to prevent division by zero

    return ndvi


def load_and_preprocess_image(img_path, num_chnl = CHANNEL):
    #ds = gdal.Open(img_path)
    #img = ds.ReadAsArray()
    #img = np.moveaxis(img, 0, -1)
    if not isinstance(img_path, str):
        img_path = img_path.decode('utf-8')
    with tifffile.TiffFile(img_path) as tif:
        img = tif.asarray()

    ndvi = compute_ndvi(img)
    ndvi = tf.expand_dims(ndvi, axis=-1)
    ndvi = (ndvi + 1.0) / 2.0
    ndvi = tf.clip_by_value(tf.cast(ndvi, tf.float32), 0., 1.)

    if (num_chnl == 3) or (num_chnl == 4):
        img = img[:,:,(3,2,1)]

    img = tf.clip_by_value(tf.cast(img, tf.float32) / MAX_VAL, 0., 1.)
    
    if num_chnl == 3:
        return img
    else:
        combined_img = tf.concat([img, ndvi], axis=-1)
        return combined_img


def load_and_preprocess_label(label_path):
    #ds = gdal.Open(label_path)
    #label = ds.ReadAsArray()
    if not isinstance(label_path, str):
        label_path = label_path.decode('utf-8')
    with tifffile.TiffFile(label_path) as tif:
        label = tif.asarray()
    label = pre_process_label(label)
    label = label[:,:,np.newaxis]
    return label


def data_generator(img_path_arr,label_path_arr):

    for (img_path,label_path) in zip(img_path_arr,label_path_arr):
        img = load_and_preprocess_image(img_path)
        label = load_and_preprocess_label(label_path)
        
        unique_vals = np.unique(label)
        
        if len(unique_vals) > 1:
            yield img, label#, sample_weights


def dataloader_gen(X_train, y_train, X_test, y_test, batch_size):
    ori_train_ds = tf.data.Dataset.from_generator(
        data_generator,
        args=(X_train, y_train),
        output_signature=(
            tf.TensorSpec(shape=(256, 256, CHANNEL), dtype=tf.float32),  # Example: Data tensor specification
            tf.TensorSpec(shape=(256,256,1), dtype=tf.uint8)
        )
    )
    # Create a dataset from the generator function
    ori_eval_ds = tf.data.Dataset.from_generator(
        data_generator,
        args=(X_test, y_test),
        output_signature=(
            tf.TensorSpec(shape=(256, 256, CHANNEL), dtype=tf.float32),  # Example: Data tensor specification
            tf.TensorSpec(shape=(256,256,1), dtype=tf.uint8) # Example: Label tensor specification
        )
    )
    augmented_train_ds = (
    ori_train_ds.shuffle(batch_size * 2)
    .batch(batch_size)
    .map(lambda img, label: (img_aug(img), tf.cast(label_aug(label),dtype=tf.uint8)), num_parallel_calls=AUTOTUNE)
    .prefetch(buffer_size=AUTOTUNE)
    )
    eval_ds = (
        ori_eval_ds
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    return augmented_train_ds, eval_ds, ori_eval_ds

############################# TRAINING MONITOR #############################

class PlotPredictions(Callback):
    def __init__(self, test_ds, num_samples=3, plot_every=5, save_path="."):
        super().__init__()
        self.test_ds = test_ds
        self.num_samples = num_samples
        self.plot_every = plot_every
        self.save_path = save_path
    
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.plot_every == 0:
            # Convert the dataset to a list
            test_ds_list = list(self.test_ds)
            fig, ax = plt.subplots(self.num_samples, 2, figsize=(10, 5 * self.num_samples))
            # Randomly select three distinct batch indices
            selected_indices = np.random.choice(len(test_ds_list), self.num_samples, replace=False)
            selected_batches = [test_ds_list[i] for i in selected_indices]

            #print(selected_batches)
            for i, batch in enumerate(selected_batches):
                img, actual_label = batch
                
                # Randomly select a sample from the current batch
                sample_index = np.random.randint(0, img.shape[0])
                img_sample = tf.reshape(img, [1, 256, 256, CHANNEL])
                actual_label_sample = actual_label

                predicted_label = np.argmax(model.predict(img_sample), axis=-1)
                predicted_label = predicted_label.reshape((predicted_label.shape[1], predicted_label.shape[2]))

                # Plot actual label
                actual_label_sample = tf.reshape(actual_label_sample, [actual_label_sample.shape[0], actual_label_sample.shape[1]])
                actual_label_sample = actual_label_sample.numpy()

                ax[i, 0].imshow(actual_label_sample, cmap='tab10', vmin=0, vmax=5)
                ax[i, 0].set_title(f"Actual Label {i+1}")

                ax[i, 1].imshow(predicted_label, cmap='tab10', vmin=0, vmax=5)
                ax[i, 1].set_title(f"Predicted Label {i+1}")

            plt.tight_layout()

            # Save the combined plot
            file_name = f"pspnet_test_{epoch}.png"
            save_location = os.path.join(self.save_path, file_name)
            plt.savefig(save_location)
            plt.close()


class ClassificationReportCallback(Callback):
    def __init__(self, model, dataset, steps, filepath="classification_report.json", matrix_path="confusion_matrix.png"):
        super(ClassificationReportCallback, self).__init__()
        self.dataset = dataset
        self.steps = steps
        self.filepath = filepath
        self.matrix_path = matrix_path
        self.reports = []

    def plot_confusion_matrix(self, cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.axis('off')
        plt.savefig(self.matrix_path)
        plt.close()

    def on_epoch_end(self, epoch, logs=None):
        if(epoch + 1) % 5 == 0:
            y_true = []
            y_pred_list = []

            # Iterate over the dataset to get predictions and true labels
            for step, (x_batch, y_batch) in enumerate(self.dataset.take(self.steps)):
                y_true.extend(y_batch.numpy().reshape(-1))
                y_pred_probs = model.predict(x_batch)
                y_pred = np.argmax(y_pred_probs, axis=-1).reshape(-1)
                y_pred_list.extend(y_pred)

            y_true = np.array(y_true)
            y_pred_list = np.array(y_pred_list)

            # Compute classification report
            report = classification_report(y_true, y_pred_list, output_dict=True)

            # Compute per-class accuracy
            accuracy_per_class = {}
            for i in range(len(np.unique(y_true))):
                correct = np.sum((y_true == i) & (y_pred_list == i))
                total = np.sum(y_true == i)
                accuracy = correct / total
                accuracy_per_class[f"class_{i}"] = accuracy

            # Compute confusion matrix
            cm = confusion_matrix(y_true, y_pred_list)
            self.plot_confusion_matrix(cm, classes=np.unique(y_true), title='Confusion Matrix')

            # Append to reports
            self.reports.append({
                "epoch": epoch,
                "classification_report": report,
                "accuracy_per_class": accuracy_per_class
            })

            # Save to JSON file
            with open(self.filepath, 'w') as f:
                json.dump(self.reports, f)



def plot_history(history, epoch_count, filename):
    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Loss at Epoch {epoch_count}')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'Accuracy at Epoch {epoch_count}')
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


############################# MODELS #############################

############## PSPNET ##############

class PSPNet(Model):
    def __init__(self, input_shape, num_classes, pretrained_weights=None):
        super(PSPNet, self).__init__()

        self.input_layer = layers.Input(shape=input_shape, name='start_input')
        self.num_classes = num_classes

        # Define the model architecture
        self.build_pspnet()
        
        if pretrained_weights:
            self.load_weights(pretrained_weights)

    def conv_block(self, X, filters, block):

        b = 'block_'+str(block)+'_'
        f1,f2,f3 = filters
        X_skip = X
        # block_a
        X = tf.keras.layers.Conv2D(filters=f1,kernel_size=(1,1),dilation_rate=(1,1),
                        padding='same',kernel_initializer='he_normal',name=b+'a')(X)
        X = tf.keras.layers.BatchNormalization(name=b+'batch_norm_a')(X)
        # X = tf.keras.layers.LeakyReLU(alpha=0.2,name=b+'leakyrelu_a')(X)
        X = tf.keras.layers.Activation("relu")(X)
        # block_b
        X = tf.keras.layers.Conv2D(filters=f2,kernel_size=(3,3),dilation_rate=(2,2),
                        padding='same',kernel_initializer='he_normal',name=b+'b')(X)
        X = tf.keras.layers.BatchNormalization(name=b+'batch_norm_b')(X)
        # X = tf.keras.layers.LeakyReLU(alpha=0.2,name=b+'leakyrelu_b')(X)
        X = tf.keras.layers.Activation("relu")(X)
        # block_c
        X = tf.keras.layers.Conv2D(filters=f3,kernel_size=(1,1),dilation_rate=(1,1),
                        padding='same',kernel_initializer='he_normal',name=b+'c')(X)
        X = tf.keras.layers.BatchNormalization(name=b+'batch_norm_c')(X)
        # skip_conv
        X_skip = tf.keras.layers.Conv2D(filters=f3,kernel_size=(3,3),padding='same',name=b+'skip_conv')(X_skip)
        X_skip = tf.keras.layers.BatchNormalization(name=b+'batch_norm_skip_conv')(X_skip)
        # block_c + skip_conv
        X = tf.keras.layers.Add(name=b+'add')([X,X_skip])
        # X = tf.keras.layers.ReLU(name=b+'relu')(X)
        X = tf.keras.layers.Activation("relu")(X)
        return X

    def base_feature_maps(self, input_layer):

        # block_1
        base = self.conv_block(input_layer,[32,32,64],'1')
        # block_2
        base = self.conv_block(base,[64,64,128],'2')
        # block_3
        base = self.conv_block(base,[128,128,256],'3')
        return base

    def pyramid_feature_maps(self, input_layer):

        base = self.base_feature_maps(input_layer) # base shape: (None, 256, 256, 256)
        # print(f'base shape:{base.shape}')
        # red
        red = tf.keras.layers.GlobalAveragePooling2D(name='red_pool')(base) # red shape: (None, 256)
        # print(f'red shape:{red.shape}')
        red = tf.keras.layers.Reshape((1,1,256))(red)
        red = tf.keras.layers.Conv2D(filters=64,kernel_size=(1,1),name='red_1_by_1')(red)
        red = tf.keras.layers.UpSampling2D(size=WIDTH,interpolation='bilinear',name='red_upsampling')(red)
        # yellow
        yellow = tf.keras.layers.AveragePooling2D(pool_size=(2,2),name='yellow_pool')(base)
        yellow = tf.keras.layers.Conv2D(filters=64,kernel_size=(1,1),name='yellow_1_by_1')(yellow)
        yellow = tf.keras.layers.UpSampling2D(size=2,interpolation='bilinear',name='yellow_upsampling')(yellow)
        # blue
        blue = tf.keras.layers.AveragePooling2D(pool_size=(4,4),name='blue_pool')(base)
        blue = tf.keras.layers.Conv2D(filters=64,kernel_size=(1,1),name='blue_1_by_1')(blue)
        blue = tf.keras.layers.UpSampling2D(size=4,interpolation='bilinear',name='blue_upsampling')(blue)
        # green
        green = tf.keras.layers.AveragePooling2D(pool_size=(8,8),name='green_pool')(base)
        green = tf.keras.layers.Conv2D(filters=64,kernel_size=(1,1),name='green_1_by_1')(green)
        green = tf.keras.layers.UpSampling2D(size=8,interpolation='bilinear',name='green_upsampling')(green)
        # base + red + yellow + blue + green
        return tf.keras.layers.concatenate([base,red,yellow,blue,green])


    def build_pspnet(self):
        # Build the PSPNet model
        X = self.pyramid_feature_maps(self.input_layer)
        X = layers.Conv2D(filters=128, kernel_size=3, padding='same', name='last_conv_3_by_3')(X)
        X = layers.BatchNormalization(name='last_conv_3_by_3_batch_norm')(X)
        output = layers.Conv2D(self.num_classes, (1, 1), activation='softmax', name='softmax_layer')(X)

        self.model = Model(inputs=self.input_layer, outputs=output)



############## UNET ##############

class UNet(Model):
    def __init__(self, input_shape, num_classes, weights_path=None):
        super(UNet, self).__init__()

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.weights_path = weights_path

        # Define the model architecture
        self.build_unet()

    def create_conv(self, input, filters, kernel_size=(3, 3), padding="same", activation="relu"):

        conv = layers.Conv2D(filters, kernel_size, padding=padding, kernel_initializer="he_normal")(input)
        conv = layers.BatchNormalization()(conv)
        conv = layers.Activation(activation)(conv)
        conv = layers.Dropout(0.2)(conv)  # Dropout layer
        conv = layers.Conv2D(filters, kernel_size, padding=padding, kernel_initializer="he_normal")(conv)
        conv = layers.BatchNormalization()(conv)
        conv = layers.Activation(activation)(conv)
        conv = layers.Dropout(0.2)(conv)  # Dropout layer
        return conv

    def create_conv_pool(self, input, filters, kernel_size=(3, 3), padding="same", activation="relu"):


        conv = self.create_conv(input, filters, kernel_size, padding, activation)
        pool = layers.MaxPooling2D(pool_size=(2, 2))(conv)
        return conv, pool
    
    def build_unet(self):

    # Input layer
        inputs = layers.Input(self.input_shape)

        # Downsampling
        conv1, pool1 = self.create_conv_pool(inputs, 64)
        conv2, pool2 = self.create_conv_pool(pool1, 128)
        conv3, pool3 = self.create_conv_pool(pool2, 256)
        conv4, pool4 = self.create_conv_pool(pool3, 512)

        conv_middle = self.create_conv(pool4, 1024)

        # Upsampling
        up7 = layers.UpSampling2D(size=(2, 2))(conv_middle)
        up7 = layers.concatenate([up7, conv4])
        conv7 = self.create_conv(up7, 512)

        up8 = layers.UpSampling2D(size=(2, 2))(conv7)
        up8 = layers.concatenate([up8, conv3])
        conv8 = self.create_conv(up8, 256)

        up9 = layers.UpSampling2D(size=(2, 2))(conv8)
        up9 = layers.concatenate([up9, conv2])
        conv9 = self.create_conv(up9, 128)

        up10 = layers.UpSampling2D(size=(2, 2))(conv9)
        up10 = layers.concatenate([up10, conv1])
        conv10 = self.create_conv(up10, 64)

        outputs = layers.Conv2D(self.num_classes, (1, 1), activation='softmax')(conv10)

        model = Model(inputs=inputs, outputs=outputs)
        if self.weights_path is not None:
            model.load_weights(self.weights_path)

        return model






############## FCN16 ##############

def bilinear(shape, dtype=None):
  
  
    filter_size = shape[0]
    num_channels = shape[2]
  
  # Convert TensorFlow dtype to NumPy dtype
    np_dtype = tf.as_dtype(dtype).as_numpy_dtype
    
    bilinear_kernel = np.zeros([filter_size, filter_size], dtype=np_dtype)
    scale_factor = (filter_size + 1) // 2
    if filter_size % 2 == 1:
        center = scale_factor - 1
    else:
        center = scale_factor - 0.5
    for x in range(filter_size):
        for y in range(filter_size):
            bilinear_kernel[x,y] = (1 - abs(x - center) / scale_factor) * \
                               (1 - abs(y - center) / scale_factor)
    weights = np.zeros((filter_size, filter_size, num_channels, num_channels))
    for i in range(num_channels):
        weights[:, :, i, i] = bilinear_kernel
  
    return weights


def FCN(n_classes = NUM_CLASSES, shape  = INPUT_SHAPE):

    Lin = layers.Input(shape=INPUT_SHAPE)
        
    Lx = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_convL1' )(Lin)
    Lx = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_convL2' )(Lx)
    Lx = layers.MaxPooling2D((2, 2), strides=(2, 2), name='blockL1_pool' )(Lx)
    f1 = Lx
    # Block 2
    Lx = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_convL1' )(Lx)
    Lx = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_convL2' )(Lx)
    Lx = layers.MaxPooling2D((2, 2), strides=(2, 2), name='blockL2_pool' )(Lx)
    f2 = Lx

    # Block 3
    Lx = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_convL1' )(Lx)
    Lx = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_convL2' )(Lx)
    Lx = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_convL3' )(Lx)
    Lx = layers.MaxPooling2D((2, 2), strides=(2, 2), name='blockL3_pool' )(Lx)
    f3 = Lx

    # Block 4
    Lx = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_convL1' )(Lx)
    Lx = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_convL2' )(Lx)
    Lx = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_convL3' )(Lx)
    Lx = layers.MaxPooling2D((2, 2), strides=(2, 2), name='blockL4_pool' )(Lx)
    f4 = Lx

    # Block 5
    Lx = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_convL1' )(Lx)
    Lx = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_convL2' )(Lx)
    Lx = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_convL3' )(Lx)
    Lx = layers.MaxPooling2D((2, 2), strides=(2, 2), name='blockL5_pool' )(Lx)
    f5 = Lx
  
    Lx = f5
    Lx = layers.Conv2D( 4096 , ( 7 , 7 ) , activation='relu' , padding='same' )(Lx)
    Lx = layers.Dropout(0.5)(Lx)
    Lx = layers.Conv2D( 4096 , ( 1 , 1 ) , activation='relu' , padding='same' )(Lx)
    Lx = layers.Dropout(0.5)(Lx)

    Lx = layers.Conv2D( n_classes ,  ( 1 , 1 ) ,activation = 'relu' )(Lx)
    Lx = layers.Conv2DTranspose( n_classes , kernel_size=(4,4) ,  strides=(2,2) , use_bias=False, kernel_initializer=bilinear )(Lx)
    Lx = layers.convolutional.Cropping2D(((1, 1), (1, 1)))(Lx)

    Lx2 = f4
    Lx2 = layers.Conv2D( n_classes ,  ( 1 , 1 ) ,activation = 'relu')(Lx2)

    Lx = layers.Add()([ Lx , Lx2 ])

    Lx = layers.Conv2DTranspose( n_classes , kernel_size=(32,32) ,  strides=(16,16) , use_bias=False )(Lx)
    Lx = layers.convolutional.Cropping2D(((8, 8), (8, 8)))(Lx)

     
    Rin = layers.Input(shape=INPUT_SHAPE)
        
    Rx = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_convR1' )(Rin)
    Rx = layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_convR2' )(Rx)
    Rx = layers.MaxPooling2D((2, 2), strides=(2, 2), name='blockR1_pool' )(Rx)
    g1 = Rx
    # Block 2
    Rx = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_convR1' )(Rx)
    Rx = layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_convR2' )(Rx)
    Rx = layers.MaxPooling2D((2, 2), strides=(2, 2), name='blockR2_pool' )(Rx)
    g2 = Rx

    # Block 3
    Rx = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_convR1' )(Rx)
    Rx = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_convR2' )(Rx)
    Rx = layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_convR3' )(Rx)
    Rx = layers.MaxPooling2D((2, 2), strides=(2, 2), name='blockR3_pool' )(Rx)
    g3 = Rx

    # Block 4
    Rx = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_convR1' )(Rx)
    Rx = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_convR2' )(Rx)
    Rx = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_convR3' )(Rx)
    Rx = layers.MaxPooling2D((2, 2), strides=(2, 2), name='blockR4_pool' )(Rx)
    g4 = Rx

    # Block 5
    Rx = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_convR1' )(Rx)
    Rx = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_convR2' )(Rx)
    Rx = layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_convR3' )(Rx)
    Rx = layers.MaxPooling2D((2, 2), strides=(2, 2), name='blockR5_pool' )(Rx)
    g5 = Rx
  
    Rx = g5
    Rx = layers.Conv2D( 4096 , ( 7 , 7 ) , activation='relu' , padding='same' )(Rx)
    Rx = layers.Dropout(0.5)(Rx)
    Rx = layers.Conv2D( 4096 , ( 1 , 1 ) , activation='relu' , padding='same' )(Rx)
    Rx = layers.Dropout(0.5)(Rx)

    Rx = layers.Conv2D( n_classes ,  ( 1 , 1 ) ,activation = 'relu' )(Rx)
    Rx = layers.Conv2DTranspose( n_classes , kernel_size=(4,4) ,  strides=(2,2) , use_bias=False, kernel_initializer=bilinear )(Rx)
    Rx = layers.convolutional.Cropping2D(((1, 1), (1, 1)))(Rx)

    Rx2 = g4
    Rx2 = layers.Conv2D( n_classes ,  ( 1 , 1 ) ,activation = 'relu')(Rx2)

    Rx = layers.Add()([ Rx , Rx2 ])

    Rx = layers.Conv2DTranspose( n_classes , kernel_size=(32,32) ,  strides=(16,16) , use_bias=False )(Rx)
    Rx = layers.convolutional.Cropping2D(((8, 8), (8, 8)))(Rx)
    
    o = layers.Add()([ Lx , Rx ])
    o = layers.Softmax(axis=3)(o)
    
    model = Model(inputs=[Lin,Rin], outputs=o)

    return model