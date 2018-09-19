import tensorflow as tf
import findspark
import pyspark as ps
from pyspark.ml.image import ImageSchema
from pyspark.sql import SparkSession
import cv2
import numpy as np


class ABO():
    '''
    Input:
        Call ABO('./Data/fruit-images-for-object-detection/test_zip/test')
    
    Output:
        N/a
    '''
    def __init__(self, path):
        self.listOfFruits = []
        #init spark
        findspark.init("C:/Users/Ryanluu2017/Desktop")

        # Build the SparkSession
        spark = SparkSession.builder \
        .master("local") \
        .appName("Linear Regression Model") \
        .config("spark.executor.memory", "1gb") \
        .getOrCreate()
        
        sc = spark.sparkContext
        RDDImages = sc.binaryFiles(path).collect()
        #format: 
        #   RDDImage[0] = path
        #   RDDImage[1] = unicode? I think

        testData = []

        for path,uni in RDDImages:
            file_bytes = np.asarray(bytearray(uni), dtype=np.uint8)
            file_bytes = cv2.imdecode(file_bytes,1)
            file_bytes = cv2.resize(file_bytes, (64,64))
            testData.append(file_bytes)

        testData = np.array(testData)
        model = tf.keras.models.load_model('./AppleBanannaOrange.h5')
        self.prediction = model.predict(testData, batch_size = 4)
        self.getFruitNames(self.prediction)

    '''
    Input:
        Numpy arr with translated images
    Outpu:  
        N/a
            Augments list of fruit names corresponding to the images 
    '''
    def getFruitNames(self, numpyArr):
        for arr in numpyArr:
            max_value = max(arr)
            max_index = np.where(arr == max_value)
            if max_index[0] == 0:
                self.listOfFruits.append('Apple')
            elif max_index[0] == 1:
                self.listOfFruits.append('Banana')
            else:
                self.listOfFruits.append('Orange')
    '''
    Input:
        N/a
    Output:
        returns the list of fruits
    '''
    def getFruitList(self):
        return self.listOfFruits

    '''
    Input:
        N/a
    Output:
        returns the list of predictions
    '''
    def getPrediction(self):
        return self.prediction


#HOW TO USE THE CLASS/FUNCTION:

if __name__ =="__main__":
    Ab = ABO('./Data/fruit-images-for-object-detection/test_zip/apple.jpg')
    print(Ab.getFruitList())
    print(Ab.getPrediction())

# #init spark
# findspark.init("/usr/local/spark")

# # Build the SparkSession
# spark = SparkSession.builder \
#    .master("local") \
#    .appName("Linear Regression Model") \
#    .config("spark.executor.memory", "1gb") \
#    .getOrCreate()
   
# sc = spark.sparkContext

# # RDDImages = sc.binaryFiles(r'./Data/fruit-images-for-object-detection/test_zip/test').take(1)
# # #RDDImages is an RDD right now, we need to change it to dataframe
# # print(type(RDDImages))
# # print(RDDImages)
# # #format: 
# # #   RDDImage[0] = path
# # #   RDDImage[1] = unicode? I think
# # file_bytes = np.asarray(bytearray(RDDImages[0][1]), dtype=np.uint8)
# # print(RDDImages[0][0])
# # R = cv2.imdecode(file_bytes,1)
# # print(R)
# # print(type(R))
# # print(R.shape)
# # prints out correct cv2 item. 

# RDDImages = sc.binaryFiles(r'./Data/fruit-images-for-object-detection/test_zip/test').collect()
# #format: 
# #   RDDImage[0] = path
# #   RDDImage[1] = unicode? I think
# testData = []
# url = []

# for path,uni in RDDImages:
#     file_bytes = np.asarray(bytearray(uni), dtype=np.uint8)
#     file_bytes = cv2.imdecode(file_bytes,1)
#     file_bytes = cv2.resize(file_bytes, (64,64))
#     testData.append(file_bytes)
#     url.append(path[-13:])

# testData = np.array(testData)
# model = tf.keras.models.load_model('./AppleBanannaOrange.h5')
# prediction = model.predict(testData, batch_size = 4)

# for item in prediction:
#     print(item)
# for url, arr in zip(url,prediction):
#     print("URL: ", url, "prediction: ", arr)
#     print(arr[0])
#     print(arr[1])
#     print(arr[2])



# images = ImageSchema.readImages(r'./Data/fruit-images-for-object-detection/test_zip/test')

# print(images.count())
# print(images.columns)
# print(images.dtypes)
# print(images.printSchema())

# import tensorflow as tf
# import os
# import numpy as np
# import cv2
# import random

# #loading model
# model = tf.keras.models.load_model('./AppleBanannaOrange.h5')

# #C:\Users\Anthoney\Documents\MachineLearning\Anthony NN\Data\fruit-images-for-object-detection\test_zip\test

# #loading test data
# imgSize = 64
# testData = []
# DATADIR = r'./Data/fruit-images-for-object-detection/test_zip/test'
# for img in os.listdir(DATADIR):
#     try:
#         if img.endswith('.jpg'):
#             img_array = cv2.imread(os.path.join(DATADIR,img), 1)
#             img_array = cv2.resize(img_array, (imgSize, imgSize))
#             testData.append([img_array, img])
#     except Exception as e:
#         pass

# img = [testData[0],testData[20],testData[40]]
# print(img[0][1])
# print(img[1][1])
# print(img[2][1])

# images = np.array([x for x,y in img])

# #evaluating model vs inputs
# prediction = model.predict(images, batch_size = 2)
# print(prediction)



# # from matplotlib import pyplot as plt
# # from random import randint
# # num = randint(0, mnist.test.images.shape[0])
# # img = mnist.test.images[num]

# # classification = sess.run(tf.argmax(y, 1), feed_dict={x: [img]})
# # plt.imshow(img.reshape(28, 28), cmap=plt.cm.binary)
# # plt.show()
# # print 'NN predicted', classification[0]