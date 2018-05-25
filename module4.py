## -*- coding: utf-8 -*-

#import os,sys,string
#import sys
#import logging
#import multiprocessing
#import time
#import json
#import cv2
#import numpy as np

#import keras
#import keras.backend as K
#from keras.datasets import mnist
#from keras.models import *
#from keras.layers import *
#from keras.optimizers import *
#from keras.callbacks import *
#from keras import backend as K

#if __name__ == '__main__':
#    x=Input([2])
#    z=Dense(1)(x)
#    model = Model(inputs=x, outputs=z)
#    model.compile(optimizer='rmsprop',
#              loss='mse')
#    X=np.array([[1,1],[2,2],[3,3]])
#    Z=np.array([[2],[4],[6]])
#    model.fit(X,Z,batch_size=1,epochs=100)  # starts training
#    q=model.predict(X)
#    K.clear_session()
   
#model.save('my_model.h5')   # HDF5 file, you have to pip3 install h5py if don't have it  
#del model  # deletes the existing model  
  
## load  
#model = load_model('my_model.h5')  
#print('test after load: ', model.predict(X_test[0:2]))  