import os
import cv2
import json
import logging
import numpy as np
import tensorflow as tf
from keras.models import load_model

class VGG16():
    def __init__(self, model_path, classes_names_path, input_size=224, gpu_memory_fraction=0.9):
        self.model_path = model_path
        self.classes_names_path = classes_names_path
        self.input_size = input_size

        # ** initialization
        self.__get_classes_name()

        # ** prepare model
        self.graph = tf.Graph()
        self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        self.sess = tf.Session(graph=self.graph, config=tf.ConfigProto(gpu_options=self.gpu_options))
        with self.sess.as_default():
            with self.graph.as_default():
                self.__load_model()
                self.image_tensor = self.graph.get_tensor_by_name('conv2d_1_input:0')
                self.scores_tensor = self.graph.get_tensor_by_name('dense_3/BiasAdd:0')

    def __get_classes_name(self):
        try:
            with open(self.classes_names_path, 'r') as f:
                self.classes_names = f.read().splitlines()
        except Exception as e:
            logging.warning(e)

    def __load_model(self):
        # ** load model and create graph
        print('** load model name : {}'.format(self.model_path))
        try:
            self.model = load_model(self.model_path)
        except Exception as e:
            print('*** failed to load model. ***')
            print(e)    

    def infer(self, image):
        # ** prepare input image
        image_np = cv2.resize(image, (self.input_size, self.input_size))
        image_np_expanded = np.expand_dims(image_np, axis=0)

        # ** infer
        feed_dict = {self.image_tensor : image_np_expanded}
        output_scores = self.sess.run(self.scores_tensor, feed_dict=feed_dict)

        return get_result_list(output_scores, self.classes_names)

    def batch_infer(self, image_list): 
        input_image_np = []

        # ** prepare input image list
        for image in image_list:
            image_np = cv2.resize(image, (self.input_size, self.input_size))
            input_image_np.append(image_np)
        image_np_expanded = np.array(input_image_np)

        # ** batch infer
        feed_dict = {self.image_tensor : image_np_expanded}
        output_scores = self.sess.run(self.scores_tensor, feed_dict=feed_dict)

        return get_result_list(output_scores, self.classes_names)

# ** enumerate all predictions
def get_result_list(output_scores, classes_names):
    result_list = []
    for score in output_scores:
        class_idx = np.argmax(score) - 1
        predict_class = classes_names[class_idx]
        res = {'class' : predict_class, 'scores': np.max(score)}
        result_list.append(res)
    return result_list