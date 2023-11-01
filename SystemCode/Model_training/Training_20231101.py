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


def training_process(train_ds, eval_ds, ori_eval_ds, epochs, batch_size, prev_epoch=None,prev_batch_size=None):


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
    plot_predictions_cb = PlotPredictions(ori_eval_ds, save_path=output_subdir_name)
    tensorboard_callback = TensorBoard(log_dir=log_dir_name, histogram_freq=1, update_freq=2)

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=1,
        mode='min',
        restore_best_weights=True
    )


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
    ds_dir = os.getcwd()
    curr_dir = os.path.join(ds_dir, 'results')
    model_chosen = 'pspnet' # or unet, fcn
    pretrained = False

    # prepare dataloaders
    X_train, X_test, y_train, y_test = train_eval_dataset_gen(ds_dir)
    augmented_train_ds, eval_ds, ori_eval_ds = dataloader_gen(X_train, y_train, X_test, y_test, BATCH_SIZE)


    model_dir_name = f'{curr_dir}/models/pspnet_{CHANNEL}_chnls_sparse_cat_loss'
    log_dir_name = f'{curr_dir}/logs/pspnet_{CHANNEL}_chnls_sparse_cat_loss'
    output_dir_name = f'{curr_dir}/output/pspnet_{CHANNEL}_chnls_sparse_cat_loss'

    os.makedirs(model_dir_name, exist_ok=True)
    os.makedirs(log_dir_name, exist_ok=True)
    os.makedirs(output_dir_name, exist_ok=True)

    # choose models
    if not pretrained: # Build model from scratch
        if model_chosen == 'pspnet':
            model = PSPNet(INPUT_SHAPE, NUM_CLASSES)
        elif model_chosen == 'unet':
            model = UNet(INPUT_SHAPE, NUM_CLASSES)
        elif model_chosen == 'fcn':
            model = FCN(shape=INPUT_SHAPE, n_classes=NUM_CLASSES)
    
    else:
        if model_chosen == 'pspnet':
            psp_model_path = f'{curr_dir}/results_2/models/pspnet_14_chnls_custom_loss/model_attention_custom_loss_bs_4_ep_100.h5'
            model = load_model(psp_model_path, custom_objects = {'loss':custom_loss})
            
        elif model_chosen == 'unet':
            unet_model_path = f'{curr_dir}/results_2/models/unet/model_custom_loss_bs_8_ep_100.h5'
            model = load_model(unet_model_path, custom_objects = {'loss':custom_loss})
            
        elif model_chosen == 'fcn':
            fcn_model_path = f'{curr_dir}/results_2/models/fcn/model_FCN16_CL.h5'    
            model = load_model(fcn_model_path, custom_objects = {'loss':custom_loss, 'bilinear':bilinear})
            

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        LEARNING_RATE, decay_steps=1000, decay_rate=0.96, staircase=True
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    loss_function = weighted_categorical_crossentropy(weights)
    model.compile(tf.keras.optimizers.Adam(LEARNING_RATE), \
                  loss=loss_function, metrics=['accuracy'])

    training_process(augmented_train_ds, eval_ds, ori_eval_ds, EPOCHS, BATCH_SIZE, prev_epoch=None,prev_batch_size=None)
