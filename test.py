import os 
import cv2
from vgg import VGG16

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

VGG16_model = VGG16(model_path = MODEL_PATH, classes_names_path = CLASSES_NAMES_PATH)

# ** predict
image = cv2.imread(IMAGE_PATH)
print(VGG16_model.infer(image))

# ** batch predict
image_list = []
for image_name in os.listdir(FOLDER_PATH):
    image = cv2.imread(os.path.join(FOLDER_PATH, image_name))
    image_list.append(image)
print(VGG16_model.batch_infer(image_list))