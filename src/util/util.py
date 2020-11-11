import cv2

def preprocess(images, req_dims=(224,224)):
    '''
        Performs necessary preprocessing to make the images in 'images' ready for input to the model. The default dims of
        req_dims=224x224 are retrieved from https://tfhub.dev/tensorflow/resnet_50/classification/1, as this is the model which
        provides feature maps to the single person parsing model.
    '''

    return [cv2.resize(img, req_dims) for img in images]

