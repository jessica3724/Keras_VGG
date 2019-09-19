import os
import cv2
import datetime
import configparser

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping

from utils.generator import data_generator
from model.vgg_model import vgg16, vgg16_keras

def main(model_config_path):
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

    # ** load model.ini
    model_config = configparser.ConfigParser()
    model_config.read(model_config_path)

    # ** data configuration
    train_set = model_config.get('data', 'train_set')
    val_set = model_config.get('data', 'val_set')

    # ** vgg16 configuration
    input_size = model_config.getint('model', 'input_size')
    num_classes = model_config.getint('model', 'num_classes')

    # ** training configuration
    epochs = model_config.getint('train', 'epochs')
    batch_size = model_config.getint('train', 'batch_size')
    # save_freq = model_config.getint('train', 'save_freq')
    learning_rate = model_config.getfloat('train', 'learning_rate')
    save_path = model_config.get('train', 'save_path')
    pretrained_path = model_config.get('train', 'pretrained_path')

    # ** GPU configuration
    os.environ['CUDA_VISIBLE_DEVICES'] = model_config.get('gpu', 'gpu')
    
    # ** set now_time folder in weight folder to model weight save path
    filename = 'ep{epoch:03d}-loss{val_loss:.3f}.h5'
    weights_directory = os.path.join(ROOT_DIR, 'weights')
    
    now_time = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    if not os.path.exists(os.path.join(ROOT_DIR, 'weights', now_time)):
        os.mkdir(os.path.join(weights_directory, now_time))
    save_path = os.path.join(weights_directory, now_time, filename)

    # ** setup keras callback
    checkpoint = ModelCheckpoint(save_path, monitor='val_loss', save_best_only=True)
    # scheduler = LearningRateScheduler(learning_rate_scheduler)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.01, patience=5, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=12, verbose=1)

    # NOTE get data information
    with open(train_set, 'r') as f:
        train_lines = f.readlines()

    with open(val_set, 'r') as f:
        val_lines = f.readlines()

    # NOTE data preprocess
    train_generator = data_generator(lines=train_lines, num_classes=num_classes, batch_size=batch_size, input_size=input_size)
    val_generator = data_generator(lines=val_lines, num_classes=num_classes, batch_size=batch_size, input_size=input_size)

    # ** train
    if pretrained_path:
        vgg16_model = __create_training_model(input_size, num_classes, learning_rate, pretrained_path)
    else:
        vgg16_model = __create_training_model(input_size, num_classes, learning_rate)
    
    vgg16_model.fit_generator(generator=train_generator, 
                              steps_per_epoch=len(train_lines)/batch_size,
                              validation_data=val_generator,
                              validation_steps=len(val_lines)/batch_size,
                              initial_epoch=0,
                              epochs=epochs,
                              callbacks=[checkpoint, reduce_lr, early_stopping],
                              verbose=1)

def __create_training_model(input_size, num_classes, learning_rate, pretrained_path=None):
    model = vgg16_keras(input_size, num_classes, pretrained_path=pretrained_path)
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learning_rate), metrics=['accuracy'])
    print('model is ready~')
    return model

if __name__ == "__main__":
    model_config_path = 'config/model.ini'
    main(model_config_path)
