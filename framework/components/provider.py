import numpy as np
import tensorflow as tf
import os
import torch.utils.data as data_utils
import torch
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_datasets as tfds

import logging

logger = logging.getLogger(__name__)

def normalise(x):
    return 2 * (x - x.min()) / (x.max() - x.min()) - 1


class SyntheticDataProvider:
    def __init__(self,path):
        logger.info('Provider loading synthetic image-label pairs.')
        data = np.load(path)
        img = data['img']
        lab = data['label']

        self.tensor = np.concatenate((img,lab),axis = 3)
        self.tensor = self.tensor.astype("float32")
        self.tensor = np.reshape(self.tensor, (-1,96, 96,2))
        self.sample = self.tensor[:10,:,:,:]
        logger.info('Synthetic image-label pairs accessed.')


    def get(self):
        logger.info('Provider creating a Tensorflow dataset with augmentation.')

        factor_train_samples = 2  # how many times to augment training samples with the ImageDataGenerator per one epoch
        rotation_range = 30  # for ImageDataGenerator
        horizontal_flip = False  # for ImageDataGenerator
        vertical_flip = True  # for ImageDataGenerator
        shear_range = 20  # for ImageDataGenerator
        width_shift_range = 0  # for ImageDataGenerator
        height_shift_range = 0  # for ImageDataGenerator

        data_gen_args = dict(rotation_range=rotation_range,
                                horizontal_flip=horizontal_flip,
                                vertical_flip=vertical_flip,
                                shear_range=shear_range,
                                width_shift_range=width_shift_range,
                                height_shift_range=height_shift_range,
                                fill_mode='constant')


        numpy_data = self.tensor
        X_datagen = ImageDataGenerator(**data_gen_args)
        y_datagen = ImageDataGenerator(**data_gen_args)

        # Provide the same seed and keyword arguments to the fit and flow methods
        seed = 1
        X  = np.reshape(numpy_data[:,:,:,0],(numpy_data.shape[0],96,96,-1))
        y = np.reshape(numpy_data[:,:,:,1],(numpy_data.shape[0],96,96,-1)) 
        
        mean_X = np.mean(X)
        std_X = np.std(X)
        X = (X-mean_X)/std_X

        X_datagen.fit(X, augment=True, seed=seed)
        y_datagen.fit(y, augment=True, seed=seed)

        X_generator = X_datagen.flow(X, batch_size=64, seed=seed, shuffle=True)
        y_generator = y_datagen.flow(y, batch_size=64, seed=seed, shuffle=True)
        
        logger.info('Augmentation done.')
        # combine generators into one which yields image and label
        return zip(X_generator, y_generator), mean_X, std_X


class ImageLabelProvider:
    def __init__(self,path):
        logger.info('Provider loading image-label pairs.')
        data = np.load(path)
        img = data['img']
        self.lab = data['label']
        self.img_norm = np.array([normalise(i) for i in img[:, :, :, :]])
        self.tensor = np.concatenate((self.img_norm,self.lab),axis = 3)
        self.tensor = np.reshape(self.tensor, (-1,96, 96,2))
        self.sample = self.tensor[:10,:,:,:]
        logger.info('Image-label data accessed.')

    def get(self, dformat = 'Dataset' ):
        
        if dformat == 'Dataset':
            logger.info('Provider creating a Tensorflow dataset.')
            return tf.data.Dataset.from_tensor_slices(self.tensor).take(-1)

        elif dformat == 'TorchDataset':
            logger.info('Provider creating a PyTorch dataset.')
            tensor_imgs = torch.FloatTensor(self.img_norm[:, :, :, 0])
            tensor_mask = torch.FloatTensor(self.lab[:, :, :, 0])
            train_pair = torch.stack((tensor_imgs, tensor_mask), 1)
            return data_utils.TensorDataset(train_pair)

        else:
            logger.info('Provider creating a numpy dataset.')
            X  = np.reshape(self.tensor[:,:,:,0],(self.tensor.shape[0],96,96,-1))
            y = np.reshape(self.tensor[:,:,:,1],(self.tensor.shape[0],96,96,-1))
            return (X,y)
        