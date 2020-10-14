import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from PIL import Image

AUTOTUNE = tf.data.experimental.AUTOTUNE

def random_image_sample(paths, chosen_images = 10):
    '''
    Displays random images from the image path
    :params
        List paths: List of image paths
        int chosen_images: number of images to display
    :return
        Void
    '''
    chosen_image_paths = np.random.choice(np.array(paths),chosen_images,replace=False)

    fig, ax = plt.subplots(1, chosen_images, figsize=(20, 2))
    for i in range(1):
        for j in range(chosen_images):
            ax[j].imshow(Image.open(chosen_image_paths[j]))
            ax[j].xaxis.set_visible(False)
            ax[j].yaxis.set_visible(False)
    fig.suptitle('Sampled Images')

def decode_img(img,img_height,img_width):
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # resize the image to the desired size
    img = tf.image.central_crop(img, 1)
    return tf.image.resize(img, [img_height, img_width])

def process_path(file_path,img_height,img_width):
    # load the raw data from the file as a string
    img = tf.io.read_file(file_path)
    img = decode_img(img,img_height,img_width)
    return img

def configure_for_performance(ds, batch_size):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size, drop_remainder=True)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


def get_image_dataset(file_pattern,img_height=180,img_width=180,batch_size=32):
    '''
    Function to return a train dataset from glob file pattern
    '''

    list_dataset = tf.data.Dataset.list_files(file_pattern, shuffle=True)
    image_count = len(list_dataset)
    list_dataset = list_dataset.shuffle(image_count, reshuffle_each_iteration=False)
    train_dataset = list_dataset.skip(0) # No validation required for GAN
    train_dataset = train_dataset.map(lambda x: process_path(x,img_height,img_width), num_parallel_calls=AUTOTUNE)
    train_dataset = configure_for_performance(train_dataset, batch_size)

    return train_dataset