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


girisverisi = np.load("animalset.npy")
girisverisi=np.reshape(girisverisi,(-1,80,80,3))
a=np.ones([210,2])*[1,0]
b=np.ones([280,2])*[0,1]
cikisverisi=np.vstack((a,b))


splitverisi=girisverisi[1:81]
splitverisi=np.append(splitverisi,girisverisi[200:280])
splitverisi=splitverisi.reshape(-1,80,80,3)
d=np.ones([80,2])*[1,0]
e=np.ones([80,2])*[0,1]
splitcikis=np.vstack((d,e))

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
model.add(Activation('softmax'))
"""

model=Sequential()
model.add(Conv2D(input_shape=(40,40,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(units=4096,activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(units=4096,activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(units=2, activation="softmax"))

"""

model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(lr=0.00001),metrics=['accuracy'])
model.summary()
history=model.fit(girisverisi,cikisverisi,batch_size=3,epochs=30,validation_data=(splitverisi,splitcikis))


print(history.history.keys())
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


model.save("animalmodel")




