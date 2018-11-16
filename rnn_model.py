import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from preprocess.preprocess import * 
from keras import regularizers
from keras.layers import GRU, LSTM, Activation, Dense, TimeDistributed , Bidirectional,RepeatVector,Input,SimpleRNN
from keras.models import Sequential,Model 
from sklearn.metrics import mean_absolute_error
from sklearn.svm import SVR

coid490 = eval(open("dataset/fund490_coid.txt").read())
df = TEJ_to_dataframe('dataset/nav_less.csv', to_weekly=False)
ind = pd.DatetimeIndex(start='2018/11/15', end='2018/11/16', freq='D')
df = pd.DataFrame(df, index=df.index.union(ind), columns=df.columns).sort_index()
df = apply_fill(df)
scaled_price = min_max(df)
rate = to_rate(df)
catgorical = pd.read_csv("dataset/merge_get_dummies.csv", encoding="big5",index_col=0)


trainX, trainy, testX, testy = train_test_split(rate,50,1,12)
wtrainX, wtestX = [], []
for i in range(len(trainX)):
    dummy_data = []
    for j in range(len(trainX[i])):
        added_data = np.hstack((trainX[i,j],catgorical.iloc[i].values))
        dummy_data.append(added_data)
    wtrainX.append(dummy_data)
wtrainX = np.array(wtrainX)

for i in range(len(testX)):
    dummy_data = []
    for j in range(len(testX[i])):
        added_data = np.hstack((testX[i,j],catgorical.iloc[i].values))

        dummy_data.append(added_data)
    wtestX.append(dummy_data)
wtestX = np.array(wtestX)

print(trainX.shape)
print(testX.shape)

'''
wtrainX = np.zeros((trainX.shape[0],trainX.shape[1],trainX.shape[2]*2))
wtestX = np.zeros((testX.shape[0],testX.shape[1],testX.shape[2]*2))
for i in range(len(trainX)):
    for k in range(len(trainX[i])):
        wtrainX[i,k] = np.hstack((trainX[i,k], wavelet_tansform(trainX[i,k])) ) 
for i in range(len(testX)):
    for k in range(len(testX[i])):
        wtestX[i,k] = np.hstack((testX[i,k], wavelet_tansform(testX[i,k])))
'''
def Rnn(trainx,trainy,testx,testy):
    model = Sequential()
    # model.add(Dropout(0.5,input_shape=(None,trainx.shape[2])))
    model.add((GRU(50,input_shape=(None,trainx.shape[2]),
            kernel_initializer='orthogonal', recurrent_initializer='orthogonal',
            return_sequences=True,
            kernel_regularizer=regularizers.l2(0.01),
            recurrent_regularizer=regularizers.l2(1e-3),
            )))
    # model.add(Bidirectional(GRU(30,
    #     kernel_initializer='orthogonal', recurrent_initializer='orthogonal',
    #     return_sequences=True,
    #     kernel_regularizer=regularizers.l2(0.0001),
    #     recurrent_regularizer=regularizers.l2(1e-5),
    #     dropout=0.3,
    #     recurrent_dropout=0.3,
    # )))
    model.add(TimeDistributed(Dense(20,activation='relu')))
    model.add(TimeDistributed(Dense(5,activation='relu')))
    model.add(Dense(1,activation='linear'))
    model.compile(loss='mae', optimizer='adam',metrics=["mae"])
    history = model.fit(trainx,trainy,
                batch_size=50,
                epochs=30,
                # shuffle=True,
                validation_data=(testx, testy),)
    y_pred = model.predict(testx).reshape(-1, 1)
    y_true = testy.reshape(-1, 1)
    mae = mean_absolute_error(y_true, y_pred)
    # model.save("GRU0.002.h5")
    print("Mae:",mae)
    return model, history, mae 
# model , history, mae =  Rnn(wtrainX, trainy, wtestX, testy)
model = SVR()
model.fit(wtrainX,trainy)
y_pred  = model.predict(wtestX).ravel()
y_true = testy.ravel()
print(mean_absolute_error(y_true,y_pred))
