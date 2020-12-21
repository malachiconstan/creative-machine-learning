import os

import tensorflow as tf
import keras
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

        if keras.__version__ == '2.2.4':
            loss = keras.losses.sparse_categorical_crossentropy
        else:
            raise NotImplementedError(f'{keras.__version__} is not supported')

        self.model.compile(
            optimizer = self.optimizer,
            loss=loss,
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

        self.model.save(os.path.join(self.checkpoint_dir,'classifier_weights.h5'))

        print('Training Completed')

    def load(self, load_h5=True):
        if load_h5:
            self.model = keras.models.load_model(os.path.join(self.checkpoint_dir,'classifier_weights.h5'))
            print('H5 Loaded')
        else:
            self.model.load_weights(os.path.join(self.checkpoint_dir,'cp.ckpt'))
            print('Checkpoint Loaded')

    def infer(self,
            infer_datadir,
            preprocessing_layer,
            img_height,
            img_width
            ):

        self.load()
        file_paths = glob(os.path.join(infer_datadir,'*.jpeg')) + glob(os.path.join(infer_datadir,'*.jpg'))
        test_pred = np.stack([preprocessing_layer(img_to_array(load_img(file, target_size=(img_height,img_width)))) for file in file_paths])

        preds = self.model.predict(test_pred)
        df_preds = pd.DataFrame(preds)
        df_preds.index = [os.path.split(fp)[1] for fp in file_paths]
        df_preds.columns = list(self.train_dataset.class_indices.keys())

        df_preds.to_excel('predictions.xlsx')

        print('Inference Completed')


def plot_image_grid(grid,
                    row_labels_left,
                    row_labels_right,
                    col_labels,
                    file_name=None,
                    figsize=None,
                    dpi=224):
    '''
    Helper function from INNvestigate library

    '''
    n_rows = len(grid)
    n_cols = len(grid[0])
    if figsize is None:
        figsize = (n_cols*3, (n_rows+1)*3)

    plt.clf()
    plt.rc("font", family="sans-serif")

    fig = plt.figure(figsize=figsize)
    fig.patch.set_facecolor('#FFFFFF')
    for r in range(n_rows):
        for c in range(n_cols):
            ax = plt.subplot2grid(shape=[n_rows+1, n_cols], loc=[r+1, c])
            # No border around subplots
            for spine in ax.spines.values():
                spine.set_visible(False)
            # TODO controlled color mapping wrt all grid entries,
            # or individually. make input param
            if grid[r][c] is not None:
                ax.imshow(grid[r][c], interpolation='none')
            else:
                for spine in plt.gca().spines.values():
                    spine.set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])

            # column labels
            if not r:
                if col_labels != []:
                    ax.set_title(col_labels[c],
                                 rotation=22.5,
                                 horizontalalignment='left',
                                 verticalalignment='bottom')

            # row labels
            if not c:
                if row_labels_left != []:
                    txt_left = [l+'\n' for l in row_labels_left[r]]
                    ax.set_ylabel(
                        ''.join(txt_left),
                        rotation=0,
                        verticalalignment='center',
                        horizontalalignment='right',
                    )

            if c == n_cols-1:
                if row_labels_right != []:
                    txt_right = [l+'\n' for l in row_labels_right[r]]
                    ax2 = ax.twinx()
                    # No border around subplots
                    for spine in ax2.spines.values():
                        spine.set_visible(False)
                    ax2.set_xticks([])
                    ax2.set_yticks([])
                    ax2.set_ylabel(
                        ''.join(txt_right),
                        rotation=0,
                        verticalalignment='center',
                        horizontalalignment='left'
                    )

    if file_name is None:
        plt.show()
    else:
        print('Saving figure to {}'.format(file_name))
        plt.savefig(file_name, orientation='landscape', dpi=dpi, bbox_inches='tight')
        plt.show()