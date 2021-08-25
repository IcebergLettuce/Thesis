import tensorflow.keras.callbacks as callbacks
import uuid

import zipfile
import os
import matplotlib.pyplot as plt
import io
import requests
import tensorflow as tf
import numpy as np
import time
import json
import traceback 
import logging

logger = logging.getLogger(__name__)


# TODO: It is not a real callback yet, but should be refactored away of the GAN implementation
class PyTorchImageGeneratorCallback:
    def __init__(self):
        pass

    def generate_image(self,epoch, gan_img, gan_label, path):
        try:
            fig, ax = plt.subplots(ncols=3, nrows=5, constrained_layout=True,figsize=(5, 10))
            fig.suptitle(f'Epoch: {epoch}', fontsize=16)

            for nrow,col in enumerate(ax):
                for ncol, axc in enumerate(col):

                    axc.set_xticklabels([])
                    axc.set_yticklabels([])
                    axc.axis('off')
                    axc.set_aspect('equal')

                    image = gan_label[nrow,:,:]
                    label = gan_img[nrow,:,:]

                    label[label>0.5] = 1.0
                    label[label<= 0.5] = 0.0

                    image =  (image - image.min()) / (image.max() - image.min())

                    if ncol == 0:
                        axc.imshow(image, cmap='gray')
                    
                    if ncol == 1:
                        axc.imshow(label, cmap='gray')

                    if ncol == 2:
                        axc.imshow(image, cmap='gray',interpolation='none')
                        masked = np.ma.masked_where(label < 0.5, label)
                        axc.imshow(masked, cmap='coolwarm_r', alpha=0.5,interpolation='none') # interpolation='none'

                    if nrow == 0:
                        if ncol ==0:
                            axc.set_title("Image")
                        if ncol ==1:
                            axc.set_title("Label")
                        if ncol ==2:
                            axc.set_title("Combined")

            img_dir = os.path.join(path,'images')
            if not (os.path.exists(img_dir)):
                os.mkdir(img_dir)

            synth_dir = os.path.join(path,'synthetic')
            if not (os.path.exists(synth_dir)):
                os.mkdir(synth_dir)

            fig.savefig(os.path.join(synth_dir,'synthetic.jpg'), bbox_inches='tight',pad_inches = 0)
            fig.savefig(os.path.join(img_dir,str(epoch)+'.jpg'), bbox_inches='tight',pad_inches = 0)
            plt.close()
        
        except Exception as e:
            print('Image Generation Failed')
            print(e)


class TrainTestCallback(callbacks.Callback):
    def __init__(self, pred_generator):
        self.pred_generator = pred_generator
    
    def on_train_end(self, logs = None):
        self.pred_generator.process(9999)


class ValidationCallback(callbacks.Callback):
    def __init__(self, pred_generator):
        self.pred_generator = pred_generator

    def on_epoch_end(self, epoch, logs=None):
        if ((epoch+1) % 30 == 0):
            logger.info('Running an evaluation on the validation dataset.')
            self.pred_generator.process(epoch)

class EvaluateSegmentationCallback(callbacks.Callback):
    def __init__(self, evaluation):
        self.evaluation = evaluation

    def on_train_end(self, logs=None):
        logger.info('Running an evaluation with on train end.')
        self.evaluation.process()

class ImageGeneratorCallback(callbacks.Callback):

    def __init__(self, file_dir, logger):
        self.file_dir = file_dir
        self.start_time = None
        self.noise = []

        for i in range(5):
            nz = tf.reshape(tf.random.uniform(shape=(1, 1,1,100)), (-1,1,1,100))
            self.noise.append(nz)

    def on_epoch_end(self, epoch, logs=None):
        self.generate_image(epoch)

    def on_epoch_begin(self, epoch, logs=None):
        logger.info(f'Epoch {epoch}')
        self.start_time = time.time()

    def generate_image(self,epoch):
        logger.info('Generate images with the Generator and store it to filesystem.')
        try:
            # Create images that are written to filesystem
            fig, ax = plt.subplots(ncols=3, nrows=5, constrained_layout=True,figsize=(5, 10))
            fig.suptitle(f'Epoch: {epoch}', fontsize=16)

            for nrow,col in enumerate(ax):
                for ncol, axc in enumerate(col):

                    axc.set_xticklabels([])
                    axc.set_yticklabels([])
                    axc.axis('off')
                    axc.set_aspect('equal')

                    noise = self.noise[nrow]

                    
                    image_label = self.model.generator(noise,training = False)

                    image_label = image_label.numpy()
                    image = image_label[0,:,:,0]
                    label = image_label[0,:,:,1]

                    label[label>0.5] = 1.0
                    label[label<= 0.5] = 0.0
    
                    image =  (image - image.min()) / (image.max() - image.min())

                    if ncol == 0:
                        axc.imshow(image, cmap='gray')
                    
                    if ncol == 1:
                        axc.imshow(label, cmap='gray')

                    if ncol == 2:
                        axc.imshow(image, cmap='gray',interpolation='none')
                        masked = np.ma.masked_where(label < 0.5, label)
                        axc.imshow(masked, cmap='coolwarm_r', alpha=0.5,interpolation='none') # interpolation='none'

                    if nrow == 0:
                        if ncol ==0:
                            axc.set_title("Image")
                        if ncol ==1:
                            axc.set_title("Label")
                        if ncol ==2:
                            axc.set_title("Combined")

            img_dir = os.path.join(self.file_dir,'images')
            if not (os.path.exists(img_dir)):
                os.mkdir(img_dir)

            fig.savefig(os.path.join(img_dir,str(epoch)+'.jpg'), bbox_inches='tight',pad_inches = 0)
            plt.close()

        except Exception as e:
            traceback.print_exc()
            logger.error(e)
    
    