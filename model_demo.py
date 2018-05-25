
## -*- coding: utf-8 -*-
##keras==2.0.5
##tensorflow==1.2.0

#import os,sys,string
#import sys
#import logging
#import multiprocessing
#import time
#import json
#import matplotlib.image as mpimg
#import numpy as np

#import keras
#import keras.backend as K
#from keras.datasets import mnist
#from keras.models import *
#from keras.layers import *
#from keras.optimizers import *
#from keras.callbacks import *
##import pydot
#from keras import backend as K
#from keras.utils import plot_model
#import cv2

#height=50
#width=150
#length=24
##识别字符集
#char_ocr='0123456789' #string.digits
##定义识别字符串的最大长度
#seq_len=12
##识别结果集合个数 0-9
#label_count=len(char_ocr)+1

#def get_label(filepath):
#    # print(str(os.path.split(filepath)[-1]).split('.')[0].split('_')[-1])
#    lab=[]
#    for num in str(os.path.split(filepath)[-1]).split('.')[0].split('_')[-1]:
#        lab.append(int(char_ocr.find(num)))
#    if len(lab) < seq_len:
#        cur_seq_len = len(lab)
#        for i in range(seq_len - cur_seq_len):
#            lab.append(label_count) #
#    return lab

##def gen_image_data(dir=r'data\train', file_list=[]):
##    dir_path = dir
##    for rt, dirs, files in os.walk(dir_path):  # =pathDir
##        for filename in files:
##            # print (filename)
##            if filename.find('.') >= 0:
##                (shotname, extension) = os.path.splitext(filename)
##                # print shotname,extension
##                if extension == '.mp4':  # extension == '.png' or
##                    file_list.append(os.path.join('%s\\%s' % (rt, filename)))
##                    # print (filename)

##    print(len(file_list))
##    index = 0
##    X = []
##    Y = []
##    for file in file_list:
##        # if index>1000:
##        #     break
##        # print(file)
##        cap=cv2.VideoCapture(file)
##        cnt=cap.get(cv2.CAP_PROP_FRAME_COUNT)
##        if cnt<length:
##            break

##        index += 1
##        temp=[]
##        cc=0
##        while(cc<length and cap.isOpened()):  
##            _, frame = cap.read()  
##            temp.append(frame[1])[:width,:height]  
##            cc+=1

##        cap.release()  
##        cv2.destroyAllWindows()
##        X.append(temp)
##        Y.append(get_label(file))

##    # print(np.shape(X))
##    X = np.array(X)
##    Y = np.array(Y)
##    return X,Y

## the actual loss calc occurs here despite it not being
## an internal Keras loss function

#def ctc_lambda_func(args):
#    y_pred, labels, input_length, label_length = args
#    # the 2 is critical here since the first couple outputs of the RNN
#    # tend to be garbage:
#    # y_pred = y_pred[:, 2:, :] 测试感觉没影响
#    y_pred = y_pred[:, :, :]
#    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

#if __name__ == '__main__':
   
#    input_tensor = Input((length,width,height,3))
#    x = input_tensor
#    x = keras.layers.convolutional.Conv3D(8,
#                                             (3,3,3), strides=(2, 1, 1),
#                                            padding='same', data_format='channels_last', 
#                                            activation='relu')(x)
#    x = MaxPooling3D(pool_size=(1,2, 2))(x)
#    for i in range(4):
#        x = keras.layers.convolutional.Conv3D(8,
#                                             (1,3,3), strides=(1,1, 1),
#                                            padding='same', data_format='channels_last', 
#                                            activation='relu')(x)
#        # x = Convolution2D(32*2**i, (3, 3), activation='relu')(x)
#        x = MaxPooling3D(pool_size=(1,2, 2))(x)
    
#    conv_shape = x.get_shape()
#    # print(conv_shape)
#    x = Reshape((int(conv_shape[1]), int(conv_shape[2]*conv_shape[3]*conv_shape[4])))(x)
#    x = Dense(10, activation='relu')(x)
#    gru_1 = GRU(15, return_sequences=True, kernel_initializer='he_normal', name='gru1')(x)
#    gru_1b = GRU(15, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1_b')(x)
#    gru1_merged = add([gru_1, gru_1b])  ###################

#    gru_2 = GRU(15, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
#    gru_2b = GRU(15, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2_b')(
#        gru1_merged)
#    x = concatenate([gru_2, gru_2b])  ######################
#    x = Dropout(0.25)(x)
#    x = Dense(label_count, kernel_initializer='he_normal', activation='softmax')(x)
#    base_model = Model(inputs=input_tensor, outputs=x)

#    labels = Input(name='the_labels', shape=[seq_len], dtype='float32')
#    input_length = Input(name='input_length', shape=[1], dtype='int64')
#    label_length = Input(name='label_length', shape=[1], dtype='int64')
#    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([x, labels, input_length, label_length])

#    model = Model(inputs=[input_tensor, labels, input_length, label_length], outputs=[loss_out])
#    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adadelta')
#    #model.summary()
#    plot_model(model, to_file='model.png',show_shapes=True)

#    #def test(base_model):
#    #    file_list = []
#    #    X, Y = gen_image_data(r'data\test', file_list)
#    #    y_pred = base_model.predict(X)
#    #    shape = y_pred[:, :, :].shape  # 2:
#    #    out = K.get_value(K.ctc_decode(y_pred[:, :, :], input_length=np.ones(shape[0]) * shape[1])[0][0])[:,
#    #          :seq_len]  # 2:
#    #    print()
#    #    error_count=0
#    #    for i in range(len(X)):
#    #        print(file_list[i])
#    #        str_src = str(os.path.split(file_list[i])[-1]).split('.')[0].split('_')[-1]
#    #        print(out[i])
#    #        str_out = ''.join([str(x) for x in out[i] if x!=-1 ])
#    #        print(str_src, str_out)
#    #        if str_src!=str_out:
#    #            error_count+=1
#    #            print('################################',error_count)
#    #        # img = cv2.imread(file_list[i])
#    #        # cv2.imshow('image', img)
#    #        # cv2.waitKey()


#    #class LossHistory(Callback):
#    #    def on_train_begin(self, logs={}):
#    #        self.losses = []

#    #    def on_epoch_end(self, epoch, logs=None):
#    #        model.save_weights('model_1018.w')
#    #        base_model.save_weights('base_model_1018.w')
#    #        test(base_model)

#    #    def on_batch_end(self, batch, logs={}):
#    #        self.losses.append(logs.get('loss'))


#    ## checkpointer = ModelCheckpoint(filepath="keras_seq2seq_1018.hdf5", verbose=1, save_best_only=True, )
#    #history = LossHistory()

#    ## base_model.load_weights('base_model_1018.w')
#    ## model.load_weights('model_1018.w')

#    #X,Y=gen_image_data()
#    cap=cv2.VideoCapture('test.mp4')
    
#    temp=[]
#    cc=0
#    while(cc<length and cap.isOpened()):  
#        _, frame = cap.read()  
#        sz=frame.shape
#        temp.append(frame[:width,:height])  
#        cc+=1

#    cap.release()  
#    cv2.destroyAllWindows()
#    X=[]
#    Y=[]
#    X.append(temp)
#    Y.append([1,2,3,4,0,0,0,0,0,0,0,0])

#    # print(np.shape(X))
#    X = np.array(X)
#    Y = np.array(Y)
#    maxin=1
#    #subseq_size = 100
#    batch_size=1
#    result=model.fit([X[:maxin], Y[:maxin], np.array(np.ones(len(X))*int(conv_shape[1]))[:maxin], np.array(np.ones(len(X))*seq_len)[:maxin]], Y[:maxin],
#                     batch_size=1,
#                     epochs=5,
#                     callbacks=[history, plotter, EarlyStopping(patience=10)], #checkpointer, history,
#                     #validation_data=([X[maxin:], Y[maxin:], np.array(np.ones(len(X))*int(conv_shape[1]))[maxin:], np.array(np.ones(len(X))*seq_len)[maxin:]], Y[maxin:]),
#                     )

#    test(base_model)

#    K.clear_session()