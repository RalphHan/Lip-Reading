import os
import numpy as np
import cv2
import keras.backend as K
import string
from keras.models import Sequential      
from keras.layers.core import Dense, Activation     
from keras.optimizers import SGD  
from keras.layers.advanced_activations import LeakyReLU  

#cap=cv2.VideoCapture(r'data\test.mp4')
#cap.set(cv2.CAP_PROP_CONVERT_RGB,False)             

##for i in range(10):
##    ret, frame = cap.read()  
##    cv2.imshow('no',mat=frame)
#while(1):
#    # get a frame
#    ret, frame = cap.read()
#    # show a frame
#    cv2.imshow("capture", frame)
#    if cv2.waitKey(100) & 0xFF == ord('q'):
#        break
#cap.release()
#cv2.destroyAllWindows() 

#a=4
model = Sequential()      
model.add(Dense(100, init='uniform', input_dim=1))      
model.add(Activation(LeakyReLU(alpha=0.01)))   
model.add(Activation('relu'))  
  