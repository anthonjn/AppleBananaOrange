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
    names = ["rottenapples", "freshapples"]
    DATADIR = r'./Data/'
    for name in names:
        path = os.path.join(DATADIR, name)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img), 1)
                img_array = cv2.resize(img_array, (imgSize, imgSize))
                if name[0] == 'r':
                    trainingData.append([img_array, 0])
                if name[0] == 'f':
                    trainingData.append([img_array, 1])
            except Exception as e:
                pass

create_training_data()
random.shuffle(trainingData)
featureSet = []
labels = []
print(len(trainingData))
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
model.add(tf.keras.layers.Dense(2, activation='softmax'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(featureSet, labels, epochs=10, batch_size=50, verbose=2)
model.save('RottenVsFreshAppleModel.h5')