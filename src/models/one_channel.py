import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import activations

class OneChannelOutputModel:
    def __init__(self, lr=0.001, input_shape=(284,284,3)):

        inputs = keras.Input(shape=input_shape)
        x = keras.applications.mobilenet_v2.preprocess_input(inputs)

        # include_top = False removes final layer from mobile net model, giving the raw feature maps
        backbone = keras.applications.mobilenet_v2.MobileNetV2(input_shape, include_top=False, weights='imagenet')
        backbone.trainable = False  # freeze backbone model
        x = backbone(x)

        for i in range(5):  # add 3 transposed convolution layers, with relu activation and batch norm
            x = keras.layers.Conv2DTranspose(512, kernel_size=(4,4), strides=(2,2))(x)
            x = keras.layers.BatchNormalization()(x)
            x = keras.layers.Activation(activations.relu)(x)

        # downscale 512 channels to 1 output channel
        outputs = keras.layers.Convolution2D(1, (3,3))(x)
        self.model = keras.Model(inputs, outputs)
        self.model.compile(optimizer=keras.optimizers.Adam(lr), loss=keras.losses.MeanSquaredError(), metrics=['accuracy'])