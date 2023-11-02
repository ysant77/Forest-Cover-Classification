# native libraries
import json
import os
from pathlib import Path

# third-party libraries
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, Sequential, Model, backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, Callback
from tensorflow.keras.models import load_model
import tifffile

# python scripts
from config import * #PSPNet, AttentionUNet, FCN

def split_channels(image, label):
    num = int(CHANNEL/2)
    first_4_channels = image[:, :, :, :num]
    last_4_channels = image[:, :, :, num:]
    return (first_4_channels, last_4_channels), label


def training_process(model_chosen, model, train_ds, eval_ds, ori_eval_ds, epochs, batch_size, prev_epoch=None,prev_batch_size=None):


    if prev_epoch is not None and prev_batch_size is not None:
        model.load_weights(f"{model_dir_name}/model_bs_{prev_batch_size}_ep_{prev_epoch}.h5")
        print(f'***Pretrained weights have been loaded***')

    # Train the model
    output_subdir_name = f'{output_dir_name}/epoch_{epochs}_bs_{batch_size}'
    plot_dir_name = f'{output_dir_name}/plots_epoch_{epochs}_bs_{batch_size}'

    os.makedirs(output_subdir_name, exist_ok=True)
    os.makedirs(plot_dir_name, exist_ok=True)

    # classification_rep_filename = f'{output_subdir_name}/classication_report_{epochs}.json'
    # confusion_matrix_filename = f'{output_subdir_name}/confusion_matrix_{epochs}.png'
    
    # prepare all the callbacks here

    model_file_name = f"{model_dir_name}/model_bs_{batch_size}_ep_{epochs}.h5"
    model_checkpoint_callback = ModelCheckpoint(
        filepath=model_file_name,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=1
    )
    plot_predictions_cb = PlotPredictions(model, ori_eval_ds, save_path=output_subdir_name)
    tensorboard_callback = TensorBoard(log_dir=log_dir_name, histogram_freq=1, update_freq=2)

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=1,
        mode='min',
        restore_best_weights=True
    )

    if model_chosen == 'fcn':
        train_ds = train_ds.map(split_channels)
        eval_ds = eval_ds.map(split_channels)
        
        history = model.fit(
            train_ds,
            validation_data=eval_ds,
            epochs=epochs,
            callbacks=[early_stopping_callback, model_checkpoint_callback],
        )

    else:
        history = model.fit(
            train_ds,
            validation_data=eval_ds,
            epochs=epochs,
            callbacks=[early_stopping_callback, model_checkpoint_callback, plot_predictions_cb, tensorboard_callback],
                    # classification_rep_callback],
            #initial_epoch=epochs - 10  # Start from the last epoch
        )




    # Plot and save the history
    filename = f"{plot_dir_name}/plots_epoch_{epochs}_bs_{batch_size}.png"
    plot_history(history, epochs, filename)

    prev_batch_size = batch_size
    prev_epoch = epochs



if __name__ == "__main__":

    # inputs to change
    ds_dir = r'D:\python_workspace\NUS_ISS\PR_Project\Dataset\DFC_Public_Dataset'
    curr_dir = os.path.join(os.getcwd(), 'results_test')
    model_chosen = 'fcn' # pspnet, unet, fcn
    pretrained = False

    # prepare dataloaders
    X_train, X_test, y_train, y_test = train_eval_dataset_gen(ds_dir)
    augmented_train_ds, eval_ds, ori_eval_ds = dataloader_gen(X_train, y_train, X_test, y_test, BATCH_SIZE)
    custom_loss = weighted_categorical_crossentropy(weights)


    model_dir_name = f'{curr_dir}/models/{model_chosen}_{CHANNEL}_chnls_test'
    log_dir_name = f'{curr_dir}/logs/{model_chosen}_{CHANNEL}_chnls_test'
    output_dir_name = f'{curr_dir}/output/{model_chosen}_{CHANNEL}_chnls_test'

    os.makedirs(model_dir_name, exist_ok=True)
    os.makedirs(log_dir_name, exist_ok=True)
    os.makedirs(output_dir_name, exist_ok=True)

    # choose models
    if not pretrained: # Build model from scratch
        if model_chosen == 'pspnet':
            # model = PSPNet(INPUT_SHAPE, NUM_CLASSES)
            model = pspnet(INPUT_SHAPE,NUM_CLASSES)
        elif model_chosen == 'unet':
            # model = AttentionUNet(INPUT_SHAPE, NUM_CLASSES)
            model = build_attention_unet(INPUT_SHAPE, num_classes=NUM_CLASSES)
        elif model_chosen == 'fcn':
            model = FCN(shape=(WIDTH, HEIGHT, int(CHANNEL/2)), n_classes=NUM_CLASSES)
    
    else:
        if model_chosen == 'pspnet':
            psp_model_path = f'{curr_dir}/models/{model_chosen}_{CHANNEL}_chnls_test/model_bs_{BATCH_SIZE}_ep_{EPOCHS}.h5'
            model = load_model(psp_model_path, custom_objects = {'loss':custom_loss})
            
        elif model_chosen == 'unet':
            unet_model_path = f'{curr_dir}/models/{model_chosen}_{CHANNEL}_chnls_test/model_bs_{BATCH_SIZE}_ep_{EPOCHS}.h5'
            model = load_model(unet_model_path, custom_objects = {'loss':custom_loss})
            
        elif model_chosen == 'fcn':
            fcn_model_path = f'{curr_dir}/models/{model_chosen}_{CHANNEL}_chnls_test/model_bs_{BATCH_SIZE}_ep_{EPOCHS}.h5'
            model = load_model(fcn_model_path, custom_objects = {'loss':custom_loss, 'bilinear':bilinear})
            

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        LEARNING_RATE, decay_steps=1000, decay_rate=0.96, staircase=True
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    model.compile(tf.keras.optimizers.Adam(LEARNING_RATE), \
                  loss=custom_loss, metrics=['accuracy'])

    training_process(model_chosen, model, augmented_train_ds, eval_ds, ori_eval_ds, EPOCHS, BATCH_SIZE, prev_epoch=None,prev_batch_size=None)
