# -*- coding:utf-8 -*-

import os,sys,string
import keras
from keras.optimizers import *
from keras.callbacks import *
from keras.models import *
from keras.layers import *
from utils import *
import keras.backend as K
import numpy as np

model= load_model('base_model.h5')
X,Y,_,_,_=gen_data(Len=24,dir_path=train_path,gray=True);
y_pred=model.predict(X)
shape = y_pred[:,2:,:].shape
ctc_decode = K.ctc_decode(y_pred[:,2:,:], 
                                  input_length=np.ones(shape[0])*shape[1])[0][0]

out = K.get_value(ctc_decode)[:, :seq_len]
for i in range(len(out)):
    pt = ''.join([char_ocr[x] for x in out[i]])
    yt = ''.join([char_ocr[x] if x!=label_count else '' for x in Y[i]])
    print(yt+'      '+pt)