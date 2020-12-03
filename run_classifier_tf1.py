import os
import numpy as np
import keras

from PIL import Image

from utils.explain import get_classifier, create_train_val, get_datasets, ClassifierTrainer

if __name__ == '__main__':
    create_train_val()

    IMG_HEIGHT = 128
    model, preprocessing_layer = get_classifier((IMG_HEIGHT,IMG_HEIGHT,3))

    def lr_schedule(epoch, lr):
        """
        Returns a custom learning rate that decreases as epochs progress.
        """
        if epoch <= 25:
            return lr
        if epoch <= 50:
            return lr/10
        if epoch <= 75:
            return lr/100
        else:
            return lr/1000

    optimiser = keras.optimizers.Adam(lr=1e-3)
    train_dataset, val_dataset = get_datasets('classifier_train_data',
                                          'classifier_val_data',
                                          preprocessing_layer,
                                          img_height=IMG_HEIGHT,
                                          img_width=IMG_HEIGHT,
                                          batch_size=32)

    trainer = ClassifierTrainer(train_dataset, val_dataset, model, optimiser, lr_schedule)
    trainer.train(epochs = 100)