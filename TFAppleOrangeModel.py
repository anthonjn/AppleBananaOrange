import tensorflow as tf
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras import optimizers
import numpy as np
import os
import cv2
import random

imgSize = 64
trainingData = []
def create_training_data():
    DATADIR = r'./Data/fruit-images-for-object-detection/train_zip/train'
    for img in os.listdir(DATADIR):
        try:
            if img.endswith('.jpg'):
                img_array = cv2.imread(os.path.join(DATADIR,img), 1)
                img_array = cv2.resize(img_array, (imgSize, imgSize))
                if img[0] == 'a':
                    trainingData.append([img_array, 0])
                elif img[0] == 'b':
                    trainingData.append([img_array, 1])
                elif img[0] == 'o':
                    trainingData.append([img_array, 2])            
        except Exception as e:
            pass

create_training_data()
random.shuffle(trainingData)
featureSet = []
labels = []

for features,label in trainingData:
    featureSet.append(features)
    labels.append(label)

featureSet = np.array(featureSet).reshape(-1, imgSize, imgSize, 3)
labels = np.array(labels)
labels = tf.keras.utils.to_categorical(labels)
#normalize the data, for pixel data, we can just / by 255, keros.normalize
#featureSet = tf.keras.utils.normalize(featureSet, axis = 1)
featureSet = featureSet/255.0

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, (5, 5), input_shape=featureSet.shape[1:], activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(3, activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(featureSet, labels, epochs=10, batch_size=5, verbose=2)
model.save('AppleBanannaOrange.h5')

# model.add(Conv2D(256, (3, 3), input_shape=featureSet.shape[1:]))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(256, (3, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

# model.add(Dense(64))

# model.add(Dense(1))
# model.add(Activation('sigmoid'))

# model.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])

# model.fit(featureSet, labels, batch_size=32, epochs=3, validation_split=0.3)








#conv2d is the convolutional layer here
# model.add(tf.keras.layers.Conv2D(32, (5,5), input_shape = featureSet.shape[1:], activation = tf.nn.relu))
# model.add(MaxPooling2D(pool_size=(2,2)))

# model.add(tf.keras.layers.Conv2D(64, (5,5), activation = tf.nn.relu))
# model.add(MaxPooling2D(pool_size=(2,2)))

# model.add(tf.keras.layers.Conv2D(128, (5,5), activation = tf.nn.relu))
# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(tf.keras.layers.Dropout(.5))

# model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(1024))

# # #output layer
# model.add(tf.keras.layers.Dense(3, activation = 'sigmoid'))
# model.compile(optimizer = 'adam' ,
#             loss = 'sparse_categorical_crossentropy',
#             metrics = ['accuracy'])

# print('waiting here')
# model.fit(featureSet, labels, batch_size = 30, epochs = 5)
# print('done')


#model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(20, activation = 'sigmoid'))
# model.add(tf.keras.layers.Dense(20, activation = tf.nn.relu))
# model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax))
# model.compile(optimizer = 'adam' ,
#             loss = 'sparse_categorical_crossentropy',
#             metrics = ['accuracy'])
# print("feature set", featureSet)
# print("labels: ", labels)
# model.fit(featureSet, labels, epochs = 3)

#to save the data
# pickle_out = open("featureSet.pickle", "wb")
# pickle.dump(featureSet, pickle_out)

# pickle_in = open("labels.pickle", "wb")
# pickle.dump(labels, pickle_in)

#to reload the model when you're done
#keras.models.load_model(filepath)