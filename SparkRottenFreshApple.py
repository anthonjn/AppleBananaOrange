import tensorflow as tf
import findspark
import pyspark as ps
from pyspark.ml.image import ImageSchema
from pyspark.sql import SparkSession
import cv2
import numpy as np

class SRFA():
    '''
    Input:
        N/A
   
    Output:
        N/A

    Notes:
        - creates the spark instance
        - sets up config 
        - SC is spark context, used for running model on node
        - setup model
    '''
    def __init__(self):
        findspark.init("/usr/local/spark")
        spark = SparkSession.builder \
            .master("local") \
            .appName("Linear Regression Model") \
            .config("spark.executor.memory", "1gb") \
            .getOrCreate()
        self.sc = spark.sparkContext
        self.model = tf.keras.models.load_model('./RottenVsFreshAppleModel.h5')
        self.model._make_predict_function()
    '''
    Input:
        path:
            - path to the image we need to find
   
    Output:
        State:
            - rotten or fresh
            - current state of fruit
    '''
    def getFruitState(self, path):
        print("error here4")
        RDDImages = self._getRDDImages(path)
        print("error here3")
        testData = self._preProcessData(RDDImages)
        print("error here2")
        prediction = self._prediction(testData)
        print("error here1")
        return prediction

    '''
    Input:
        testData:
            - array of preprocessed data
   
    Output:
        listOfFruit: 
            - returns whether it's rotten or fresh
    '''
    def _prediction(self, testData):
        print('err here')

        predict = self.model.predict(testData, batch_size=4)
        print('err here2')

        listOfFruit = []
        for val in predict:
            max_value = max(val)
            max_index = np.where(val == max_value)
            if max_index[0] == 0:
                listOfFruit.append('Rotten')
            elif max_index[0] == 1:
                listOfFruit.append('Fresh')
        return listOfFruit

    '''
    Input:
        RDDImages:
            - array of binary files from path
   
    Output:
        testData:
            - np array, preprocesed by resizing ect.
    '''
    def _preProcessData(self, RDDImages):
        testData = []
        for path,uni in RDDImages:
            file_bytes = np.asarray(bytearray(uni), dtype=np.uint8)
            file_bytes = cv2.imdecode(file_bytes,1)
            file_bytes = cv2.resize(file_bytes, (64,64))
            testData.append(file_bytes)
        return np.array(testData)

    '''
    Input:
        path:
            - path to the image we need to find
   
    Output:
        array of binary files from the path
            format: 
                RDDImage[0] = path
                RDDImage[1] = unicode
    '''
    def _getRDDImages(self, path):
        return self.sc.binaryFiles(path).collect()

# How to run this:
# ab = SRFA()
# print(ab.getFruitState('./Data/fruit-images-for-object-detection/test_zip/apple.jpg'))