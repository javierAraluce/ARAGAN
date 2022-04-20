import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import cv2
from config import image_input_size, batch_size, imgs_train_path, maps_train_path, imgs_val_path, maps_val_path, mask_input_size
from tqdm import tqdm



def data_generator():
    '''
    Create data generator to fit the model 
    '''
    datagen_args = dict(rescale=1/255.,
                        samplewise_center=True, 
                        samplewise_std_normalization=True)
    
    image_train_generator = ImageDataGenerator(**datagen_args)
    image_val_generator = ImageDataGenerator(**datagen_args)
    mask_train_generator = ImageDataGenerator(**datagen_args)
    mask_val_generator = ImageDataGenerator(**datagen_args)

    nb_images_train = (len(os.listdir(imgs_train_path + 'all_images/')))
    nb_images_val = (len(os.listdir(imgs_val_path + 'all_images/')))


                
    images_train_generator = image_train_generator.flow_from_directory(imgs_train_path, 
        target_size=image_input_size[0:2], 
        batch_size = batch_size, class_mode=None)
    masks_train_generator = mask_train_generator.flow_from_directory(maps_train_path, 
        target_size=mask_input_size[0:2], color_mode= 'grayscale',
        batch_size = batch_size, class_mode=None)
    train_generator = zip(images_train_generator, masks_train_generator)


    images_val_generator = image_val_generator.flow_from_directory(
        imgs_val_path, target_size=image_input_size[0:2], 
        batch_size = batch_size, class_mode=None)
    masks_val_generator = mask_val_generator.flow_from_directory(
        maps_val_path, target_size=mask_input_size[0:2], color_mode= 'grayscale',
        batch_size = batch_size, class_mode=None)

    val_generator = zip(images_val_generator, masks_val_generator)

               

    return train_generator, val_generator, nb_images_train, nb_images_val