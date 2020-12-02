import os

import tensorflow as tf
import keras
import datetime as dt
import numpy as np
import pandas as pd

from glob import glob
from shutil import copyfile
from PIL import Image
from keras.preprocessing.image import img_to_array, load_img

def get_classifier(input_shape, num_classes=19):
    base_model = keras.applications.DenseNet169(input_shape=input_shape, include_top=False, weights='imagenet')
    preprocessing_layer = keras.applications.densenet.preprocess_input
    global_average_layer = keras.layers.GlobalAveragePooling2D()
    prediction_layer = keras.layers.Dense(num_classes, activation="softmax")
    
    x = base_model.output
    x = global_average_layer(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = keras.Model(base_model.input, outputs)
    return model, preprocessing_layer

def create_train_val(data_dir = 'classifier_data', train_percent = 0.8):
    cwd = os.getcwd()
    assert 'classifier_train_data' not in cwd, 'classifier_train_data already created'
    assert 'classifier_val_data' not in cwd, 'classifier_val_data already created'
    
    directories = glob(os.path.join(cwd, data_dir + '/*'))
    
    for directory in directories:
        file_paths = np.array(os.listdir(directory))
        mask = np.random.choice((True, False), size = file_paths.shape[0], p = (train_percent, 1-train_percent))
        train_file_paths = file_paths[mask]
        val_file_paths = file_paths[np.invert(mask)]
        class_name = os.path.basename(directory)
        for picture_file_name in train_file_paths:
            destination_directory = os.path.join(cwd,'classifier_train_data',class_name)
            if not os.path.exists(destination_directory):
                os.makedirs(destination_directory)
            
            destination = os.path.join(destination_directory,picture_file_name)
            if not os.path.exists(destination):
                copyfile(os.path.join(directory,picture_file_name), destination)

        for picture_file_name in val_file_paths:
            destination_directory = os.path.join(cwd,'classifier_val_data',class_name)
            if not os.path.exists(destination_directory):
                os.makedirs(destination_directory)
            
            destination = os.path.join(destination_directory,picture_file_name)
            if not os.path.exists(destination):
                copyfile(os.path.join(directory,picture_file_name), destination)
        
        print(f'Completed for {class_name}')

    print('Completed Generation of Train and Val Sets')

def get_datasets(train_data_dir, val_data_dir, preprocessing_layer,img_height=180, img_width=180, batch_size=32):

    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        preprocessing_function=preprocessing_layer,
    )

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='sparse')

    val_datagen = keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocessing_layer,
    )

    val_generator = val_datagen.flow_from_directory(
        val_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        shuffle=False,
        class_mode='sparse')

    return train_generator, val_generator

class ClassifierTrainer(object):
    '''
    Trainer class for classifier. For keras 2.2.4 and tensorflow 1.x
    '''
    def __init__(self,
                train_dataset,
                validation_dataset,
                model,
                optimizer,
                lr_schedule
                ):
        '''
        __init__ method. Instantiates logs and checkpoint directories, as well as relevant callbacks to be used for training

        :params:
        keras.Dataset train_dataset: Training dataset
        keras.Dataset validation_dataset: Validation dataset
        keras.Model model: Classifier Model. Typically one pre-trained on Imagenet
        keras.Optimizer optimizer: Optimizer used for training e.g. Adam
        function lr_schedule: A function that takes in an epoch and returns the relevant learning rate
        '''

        # Define Directories
        current_time = dt.datetime.now().strftime("%Y%m%d-%H%M")
        self.log_dir = os.path.join(os.getcwd(),'classifier_1x_logs',current_time)
        self.checkpoint_dir = os.path.join(os.getcwd(),'classifier_1x_checkpoints')

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.tensorboard_callback = keras.callbacks.TensorBoard(log_dir=self.log_dir)

        # Define Checkpoint
        self.cp_callback = keras.callbacks.ModelCheckpoint(filepath=os.path.join(self.checkpoint_dir,'cp.ckpt'), verbose=1, save_weights_only=True)

        self.model = model
        self.optimizer = optimizer

        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset

        # Define Learning Rate Scheduler
        self.lr_callback = keras.callbacks.LearningRateScheduler(lr_schedule)

        self.model.compile(
            optimizer = self.optimizer,
            loss=keras.losses.sparse_categorical_crossentropy,
            metrics = ["accuracy"]
        )
    
    def train(self,
            epochs=10,
            batch_size=32
            ):
        
        self.history = self.model.fit_generator(
            self.train_dataset,
            steps_per_epoch=len(self.train_dataset),
            epochs=epochs,
            validation_data=self.validation_dataset,
            validation_steps=len(self.validation_dataset),
            callbacks=[self.cp_callback, self.tensorboard_callback, self.lr_callback],
        )

        print('Training Completed')

    def infer(self,
            infer_datadir,
            preprocessing_layer,
            img_height,
            img_width
            ):

        # self.model.load_weights(os.path.join(self.checkpoint_dir,'cp.ckpt'))
        file_paths = glob(os.path.join(infer_datadir,'*.jpeg')) + glob(os.path.join(infer_datadir,'*.jpg'))
        test_pred = np.stack([preprocessing_layer(img_to_array(load_img(file, target_size=(img_height,img_width)))) for file in file_paths])

        preds = self.model.predict(test_pred)
        df_preds = pd.DataFrame(preds)
        df_preds.index = [os.path.split(fp)[1] for fp in file_paths]
        df_preds.columns = list(self.train_dataset.class_indices.keys())

        df_preds.to_csv('predictions.csv')

        print('Inference Completed')