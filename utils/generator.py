import os 
import cv2
import random
import numpy as np
from keras.utils import to_categorical

def data_generator(lines, num_classes, batch_size, input_size):
    idx = len(lines) - 1

    # NOTE shuffle
    random.shuffle(lines)

    while True:
        image_data_list = []
        class_num_list = []

        # NOTE create generator
        for i in range(batch_size):
            if idx < 0:
                idx = len(lines) - 1

                # NOTE shuffle
                random.shuffle(lines)

            image_path, class_num = lines[idx].split(' ')
            image = cv2.imread(image_path)
            # NOTE resize with padding
            height, width, channels = image.shape
            image = cv2.resize(image, (input_size, input_size)) # (width, height)
            image = image / 255.
            
            # NOTE num_classes=2 labels as 'one hot'
            class_one_hot = to_categorical(int(class_num), num_classes=num_classes)

            image_data_list.append(image)
            class_num_list.append(class_one_hot)            
            idx -= 1
        image_data_list = np.array(image_data_list) # (2, 224, 224, 3) => (batch_size, height, width, channels)
        class_num_list = np.array(class_num_list) # (3, 2) => (batch_size, num_classes)
        yield (image_data_list, class_num_list)
