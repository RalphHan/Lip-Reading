
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
from keras.utils import plot_model

if __name__ == "__main__":
    autoencoder= load_model('autoencoder.h5')
    plot_model(autoencoder, to_file='autoencoder.png',show_shapes=True)
    input_tensor =autoencoder.input
    x=autoencoder.get_layer('encode').output
    del autoencoder
    x=Dropout(0.5,name='dp1')(x)

    gru_1 = GRU(32, return_sequences=True, kernel_initializer='he_normal', name='gru1')(x)
    gru_1b = GRU(32, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(x)
    gru1_merged = add([gru_1, gru_1b])
    gru1_merged=Dropout(0.5,name='dp2')(gru1_merged)

    gru_2 = GRU(32, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
    gru_2b = GRU(32, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(
        gru1_merged)
    x = concatenate([gru_2, gru_2b])
    x = Dropout(0.25,name='dp3')(x)
    x = Dense(label_count, kernel_initializer='he_normal', activation='softmax',name='base_model')(x)
    base_model = Model(inputs=input_tensor, outputs=x)

    labels = Input(name='the_labels', shape=[seq_len], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([x, labels, input_length, label_length])

    model = Model(inputs=[input_tensor, labels, input_length, label_length], outputs=[loss_out])
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adadelta')
    #model.load_weights('model2.w',by_name=True)
    model.summary()
    plot_model(model, to_file='model.png',show_shapes=True)
    class LossHistory(Callback):
        def __init__(self):    
            super(Callback,self).__init__()  #使用super函数  
            self.cnt=1   
        def on_epoch_end(self, epoch, logs=None):
            if self.cnt%100==0:
                base_model.save('base_model.h5')
                model.save_weights('model.w')
            self.cnt+=1

    history = LossHistory()
    
    X=gen_data(Len=24,gray=True);
    X_V=gen_data(Len=24,dir_path=valid_path,gray=True);
    model.fit(X[:4],X[4],
                     batch_size=30,
                     epochs=1000,
                     callbacks=[history],#, EarlyStopping(patience=10)
                     validation_data=(X_V[:4],X_V[4])
                     )
    #test(base_model)
    base_model.save('base_model.h5')
    model.save_weights('model.w')
    K.clear_session()