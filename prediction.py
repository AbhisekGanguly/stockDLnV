import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np 
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

def predict_stock(stock_symbol):
  data = yf.download(tickers=stock_symbol, period='5y', interval='1d')
  opn = data[['Open']]
  ds = opn.values
  normalizer = MinMaxScaler(feature_range=(0,1))
  ds_scaled = normalizer.fit_transform(np.array(ds).reshape(-1,1))
  train = int(len(ds_scaled)*0.70)
  test = int(len(ds_scaled) - train)
  ds_train, ds_test = ds_scaled[0:train, :], ds_scaled[train:len(ds_scaled), :1]

  #Making timeseries model
  def create_ds(dataset, step):
    XTrain, YTrain = [], []
    for i in range(len(dataset)-step-1):
      a = dataset[i:(i+step), 0]
      XTrain.append(a)
      YTrain.append(dataset[i+step, 0])
    return np.array(XTrain), np.array(YTrain)

  timestamp = 100
  X_train, Y_train = create_ds(ds_train, timestamp)
  X_test, Y_test = create_ds(ds_test, timestamp)

  X_train = X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
  X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

  model = Sequential()
  model.add(LSTM(units=50,return_sequences=True, input_shape=(X_train.shape[1],1)))
  model.add(LSTM(units=50,return_sequences=True))
  model.add(LSTM(units=50) )
  model.add(Dense(units=1,activation='linear'))
  model.summary()

  model.compile(loss = 'mean_squared_error', optimizer='adam')
  model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=100, batch_size=64)

  train_predict = model.predict(X_train)
  test_predict = model.predict(X_test)
  train_predict = normalizer.inverse_transform(train_predict)
  test_predict = normalizer.inverse_transform(test_predict)

  fut_inp = ds_test[len(ds_test)-100:]
  fut_inp = fut_inp.reshape(1,-1)
  tmp_inp = list(fut_inp)
  tmp_inp = tmp_inp[0].tolist()

  lst_output = []
  n_steps = 100
  i = 0
   
  while(i<30):

    if(len(tmp_inp)>100):
      fut_inp = np.array(tmp_inp[1:])
      fut_inp=fut_inp.reshape(1,-1)
      fut_inp = fut_inp.reshape((1, n_steps, 1))
      yhat = model.predict(fut_inp, verbose=0)
      tmp_inp.extend(yhat[0].tolist())
      tmp_inp = tmp_inp[1:]
      lst_output .extend(yhat.tolist())
      i = i+1
 
    else:
      fut_inp = fut_inp.reshape((1, n_steps,1))
      yhat = model.predict(fut_inp, verbose=0)
      tmp_inp.extend(yhat[0].tolist())
      lst_output.extend(yhat.tolist())
      i = i+1

  plot_new = np.arange(1,101)
  plot_pred = np.arange(101, 131)
  ds_new = ds_scaled.tolist()

  ds_new.extend(lst_output)

  final_graph = normalizer.inverse_transform(ds_new).tolist()

  return("Price after 30D {0}".format(round(float(*final_graph[len(final_graph)-1]),2)))