import keras_preprocessing
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import keras
from keras.layers import Dense, Activation, Conv2D, Dropout, MaxPooling2D
from keras.models import Sequential
from keras import *
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation, Flatten,Dense,Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D,ZeroPadding2D,AveragePooling2D
from keras.layers import Input,Lambda, Dense, Flatten
from keras.models import Model
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping

model=Sequential()
model.add(Conv2D(64,3,input_shape=(80,80,3)))
model.add(Conv2D(64,2))
model.add(Conv2D(64,2))
model.add(Conv2D(64,2))
model.add(MaxPooling2D((3,3)))
model.add(Conv2D(64,2))
model.add(Conv2D(64,2))
model.add(Conv2D(64,2))
model.add(MaxPooling2D((3,3)))
model.add(Conv2D(64,2))
model.add(Conv2D(64,2))
model.add(Conv2D(64,2))
model.add(MaxPooling2D((2,2)))
model.add(Conv2D(64,2))
model.add(Flatten())
model.add(Dense(1000,activation='relu')) #alexnet
model.add(Dropout(0.3))
model.add(Dense(1000,activation='relu')) #alexnet
model.add(Dropout(0.3))
model.add(Dense(2))#ya kapalı ya açık
model.add(Activation('softmax')) #sonkatman
model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(lr=0.00001),metrics=['accuracy'])

model.load_weights("animalmodel")

boyutlandirilmisresim=cv2.imread("test4.jpg")
girisverisi = np.array([])
testimage = cv2.resize(boyutlandirilmisresim, (80, 80))
girisverisi = np.append(girisverisi, testimage)
girisverisi = np.reshape(girisverisi, (-1, 80, 80,3))
cikisverisi = np.array([])
cikisverisi = model.predict(girisverisi)

testimage = np.array([0.9, 0])
comparison = cikisverisi  >= testimage


esitlik = comparison.all()


if esitlik == True:
               print("sheep")

else:
               print("cow")