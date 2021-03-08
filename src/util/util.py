import cv2
import os

def preprocess(images, req_dims=(224,224)):
    '''
        Performs necessary preprocessing to make the images in 'images' ready for input to the model. The default dims of
        req_dims=224x224 are retrieved from https://tfhub.dev/tensorflow/resnet_50/classification/1, as this is the model which
        provides feature maps to the single person parsing model.
    '''

    return [cv2.resize(img, req_dims) for img in images]

def get_paths(input_dir, target_dir):

    '''
        Returns the paths of all valid inputs / targets in the specified directories. Validity of a given
        target is determined by checking if a corresponding input file appears in the input directory, and vice-versa.
    '''

    # read input and target files, ensuring that each input file has a corresponding target file in target_dir
    avail_target_filenames = {os.path.splitext(filename)[0] for filename in os.listdir(target_dir)}
    valid_filenames = [filename for filename in os.listdir(input_dir) if os.path.splitext(filename)[0] in avail_target_filenames]
    input_paths = [input_dir + filename + '.jpg' for filename in valid_filenames]
    target_paths = [target_dir + filename + '.png' for filename in valid_filenames]

    print(f'File paths successfully loaded - {len(avail_target_filenames) - len(valid_filenames)} path(s) ignored.')

    return input_paths, target_paths


