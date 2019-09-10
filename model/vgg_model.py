from keras.models import Sequential
from keras.models import Model

from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D

from keras.applications.vgg16 import VGG16

def vgg16(input_size, num_classes, pretrained_path=None):
    model = Sequential()

    model.add(Conv2D(64, (3, 3), input_shape=(input_size, input_size, 3), activation='relu', padding='same', trainable=False))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', trainable=False))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', trainable=False))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', trainable=False))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', trainable=False))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', trainable=False))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', trainable=False))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', trainable=False))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', trainable=False))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', trainable=False))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', trainable=False))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', trainable=False))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', trainable=False))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model.add(Flatten())

    if num_classes==2:
        activation = 'sigmoid'
    else:
        activation = 'softmax'

    model.add(Dense(8, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(num_classes, activation=activation))

    if pretrained_path:
        model.load_weights(pretrained_path)
    print(model.summary())
    return Model(inputs=model.input, outputs=model.output)
	
def vgg19(input_size, num_classes):
    model = Sequential()
    
    model.add(Conv2D(64, (3, 3), input_shape=(input_size, input_size, 3), activation='relu', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())

    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    print(model.summary())
    return model