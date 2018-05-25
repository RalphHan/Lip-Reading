import os
import numpy as np
import cv2
import keras.backend as K
import string
height=128
width=160
INF=0x3f3f3f3f
#识别字符集
char_ocr='abcdefghijklmnopqrstuvwxyz' #string.digits
#定义识别字符串的最大长度
seq_len=6
#识别结果集合个数 0-9
label_count=len(char_ocr)+1
train_path=r'data\train'
valid_path=r'data\validation'
test_path=r'data\test'
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    # y_pred = y_pred[:, 2:, :] 测试感觉没影响
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
def get_label(str):
    # print(str(os.path.split(filepath)[-1]).split('.')[0].split('_')[-1])
    lab=[]
    for c in str:
        lab.append(int(char_ocr.find(c)))
    if len(lab) < seq_len:
        cur_seq_len = len(lab)
        for i in range(seq_len - cur_seq_len):
            lab.append(label_count) #
    return lab
class getoutofloop(Exception): pass
def gen_data(dir_path=train_path,Len=INF,tot=INF,gray=False):
    X = []
    Y = []
    L=[]
    S=[]
    index=0
    if Len&1:
        Len-=1
    try:
        for rt, dirs, files in os.walk(dir_path):  # =pathDir
            for filename in files:
                if not index<tot:
                    raise getoutofloop()
                shotname,_= os.path.splitext(filename)
                file=os.path.join('%s\\%s' % (rt, filename))
                cap=cv2.VideoCapture(file)
                cnt=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                ht=max(0,height-int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                wd=max(0,width-int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
                if cnt>Len:
                    cnt=Len
                temp=[]
                cc=0
                while(cc<cnt and cap.isOpened()):  
                    _, frame = cap.read() 
                    frame=cv2.copyMakeBorder(frame[:height,:width],0,ht,0,wd,cv2.BORDER_REPLICATE)
                    if gray:
                        frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).reshape((height,width,1))
                    temp.append(frame) 
                    cc+=1

                for i in range(Len-cnt):
                    temp.append(frame)

                cap.release()  
                cv2.destroyAllWindows()
                X.append(temp)
                Y.append(get_label(shotname.lower()))
                L.append([(cnt>>1)-2])
                S.append([len(shotname)])
                index+=1
    except getoutofloop:
        pass
    X = np.array(X)
    Y = np.array(Y)
    L=np.array(L)
    S=np.array(S)
    A=np.ones(len(X)).reshape((-1,1))
    return [X,Y,L,S,A]