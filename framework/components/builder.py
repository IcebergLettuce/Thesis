import tensorflow as tf
from tensorflow import keras

tf.get_logger().setLevel('INFO')

from .architectures import get_unet, DPWGAN, MINIGAN

def dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    interesection = tf.reduce_sum(y_true * y_pred)
    denominator = tf.reduce_sum(y_true + y_pred)
    return (-1)*(2.0 * interesection + 1) / (denominator + 1)

class ModelBuilder:
    @staticmethod
    def build(hp):

        if(hp['architecture']=='UNET'):
            model = get_unet(96,1,'relu','sigmoid',keras.optimizers.Adam,hp['learning_rate'],hp['dropout_rate'],dice_loss, metrics=['accuracy'])
            return model

        elif(hp['architecture']=='DPWGAN'):
            model = DPWGAN(hp)
            print(model)
            print(hp)
            return model

        elif(hp['architecture']=='MINIGAN'):
            model = MINIGAN(hp)
            return model
        
        else:
            raise Exception(f'model architecture {hp["architecture"]} not registered.')

        return model
