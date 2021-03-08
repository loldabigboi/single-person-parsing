import cv2
import os
import numpy.random as rand

class DataLoader:
    def __init__(self, img_path, seg_path):
        self.img_path = img_path
        self.seg_path = seg_path

    def read_data(self, n=32, seg=False, shuffle=True):
        '''
            Reads 'n' images from img_path (or seg_path if seg=True). If n == -1 , all images are read. If shuffle is
            True (default), then the images are also read in a random order.
        '''

        path = self.img_path if not seg else self.seg_path

        file_names = os.listdir(path)
        rand.shuffle(file_names)
        
        img_dict = {}
        for i, file_name in enumerate(file_names):

            if (i == n):  # reached required number of read images
                break
            
            ext = os.path.splitext(file_name)[1]
            if (ext not in ('.jpg', '.jpeg', '.png')):  # only read jpg & png images
                print(f'Skipping file \'{file_name}\' as its extension is not supported as an image file format.')
            else:
                img = cv2.imread(path + file_name, cv2.COLOR_BGR2RGB)  # read image as np ndarray
                if len(img.shape) == 2:  # convert grayscale to rgb
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                img_dict[file_name] = img
        
        return img_dict

    def read_paths(self, paths):
        '''
            Reads and returns an image for each path in 'paths' as a list.
        '''
        
        images = []
        for path in enumerate(paths):
            
            img = cv2.imread(path, cv2.COLOR_BGR2RGB)  # read image as np ndarray
            if len(img.shape) == 2:  # convert grayscale to rgb
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            images.append(img)
        
        return images
