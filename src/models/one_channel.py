from src.util.data_loader import DataLoader
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import activations

class OneChannelOutputModel(keras.Model):
    def __init__(self, input_shape=(224,224,3)):

        inputs = keras.Input(shape=input_shape)
        self.preprocess_input = keras.applications.mobilenet_v2.preprocess_input
        x = self.preprocess_input(inputs)

        # include_top = False removes final layer from mobile net model, giving the raw feature maps
        backbone = keras.applications.mobilenet_v2.MobileNetV2(input_shape, include_top=False, weights='imagenet')
        backbone.trainable = False  # freeze backbone model
        x = backbone(x)

        for i in range(5):  # add 3 transposed convolution layers, with relu activation and batch norm
            x = keras.layers.Conv2DTranspose(512, kernel_size=(4,4), strides=(2,2))(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Activation(activations.relu)(x)

        # downscale 512 channels to 1 output channel
        self.outputs = keras.layers.Convolution2D(1, (3,3))(x)
        super(OneChannelOutputModel, self).__init__(inputs=inputs, outputs=self.outputs)

    def train_step(self, data):

        dl = DataLoader('','')

        # read images from file paths
        input_paths, output_paths = data
        x = dl.read_paths(input_paths)
        # need to resize targets to match size of output layer
        y = [cv2.resize(img, self.outputs.get_shape()[:2]) for img in dl.read_paths(output_paths)]

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            # compute loss
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # compute gradients
        gradients = tape.gradient(loss, self.trainable_variables)
        # update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        # update metrics
        self.compiled_metrics.update_state(y, y_pred)

        return {m.name : m.result() for m in self.metrics}

