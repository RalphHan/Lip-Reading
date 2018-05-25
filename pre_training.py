# -*- coding: utf-8 -*-
import os,sys,string
import cv2
import numpy as np
from utils import *
import keras
import keras.backend as K
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import *

if __name__ == "__main__":
    #encoder
    input_tensor = Input((None,height,width,1))
    x = input_tensor
    x = keras.layers.convolutional.Conv3D(8,
                                             (3,3,3), strides=(2, 1, 1),
                                            padding='same', data_format='channels_last', 
                                            activation='relu')(x)
    x = MaxPooling3D(pool_size=(1,2, 2))(x)
    x=Dropout(0.5)(x)

    for i in range(4):
        x = keras.layers.convolutional.Conv3D(16,
                                             (1,3,3), strides=(1,1, 1),
                                            padding='same', data_format='channels_last', 
                                            activation='relu')(x)
        # x = Convolution2D(32*2**i, (3, 3), activation='relu')(x)
        x = MaxPooling3D(pool_size=(1,2, 2))(x)
        if i!=3:
            x=Dropout(0.5)(x)
    
    conv_shape = x.get_shape()
    # print(conv_shape)
    x = Reshape((-1, int(conv_shape[2]*conv_shape[3]*conv_shape[4])))(x)
    x=Dropout(0.5)(x)
    encode =Dense(32,name='encode')(x)
    base_model = Model(inputs=input_tensor, outputs=encode)
    #decoder
    x=Dropout(0.5)(encode)
    x =Dense(int(conv_shape[2]*conv_shape[3]*conv_shape[4]), activation='relu')(x)
    x = Reshape((-1, int(conv_shape[2]),int(conv_shape[3]),int(conv_shape[4])))(x)

    for i in range(3):
        x = UpSampling3D((1,2, 2))(x)
        x=Dropout(0.5)(x)
        x = keras.layers.convolutional.Conv3D(16,
                                             (1,3,3), strides=(1,1, 1),
                                            padding='same', data_format='channels_last', 
                                            activation='relu')(x)

    x = UpSampling3D((1,2, 2))(x)
    x=Dropout(0.5)(x)
    x = keras.layers.convolutional.Conv3D(8,
                                         (1,3,3), strides=(1,1, 1),
                                         padding='same', data_format='channels_last', 
                                         activation='relu')(x)
    x = UpSampling3D((2,2, 2))(x)
    x=Dropout(0.5)(x)
    decoded = keras.layers.convolutional.Conv3D(1,
                                             (3,3,3), strides=(1,1, 1),
                                            padding='same', data_format='channels_last', 
                                            activation='relu')(x)
    autoencoder = Model(input_tensor, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    X,_,_,_,_=gen_data(Len=24,gray=True);
    X_V,_,_,_,_=gen_data(dir_path=valid_path,Len=24,gray=True);
    class LossHistory(Callback):
        def on_epoch_end(self, epoch, logs=None):
            autoencoder.save('autoencoder.h5')
            #test(base_model)

    history = LossHistory()

    autoencoder.fit(X,X,batch_size=20,epochs=200,
                   callbacks=[history, EarlyStopping(patience=10)],
                   validation_data=(X_V,X_V))
# 得到编码层的输出
#autoencoder= load_model('autoencoder.h5')
#encoder_model = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encode').output)