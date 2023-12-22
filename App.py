import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt

model = load_model(r'Stock Market Prediction 44 Moving Average.keras')

st.header('STOCK MARKET PREDICTOR')
stock = st.text_input('Enter Stock Symbol' , 'GOOG')
start = '2010-01-01'
end = '2023-12-20'

data = yf.download(stock , start , end )

st.subheader('stock data')
st.write(data)


data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80) :])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

pas_44days = data_train.tail(44)
data_test = pd.concat([pas_44days , data_test] , ignore_index = True)
data_test_scale = scaler.fit_transform(data_test)

st.subheader(' Price vs MA 44')
ma_44days = data.Close.rolling(44).mean()

fig1=plt.figure(figsize=(8,6))
plt.plot(ma_44days,'r', label ='MA44')
plt.plot(data.Close , 'g', label ='Closing Price')
plt.legend()
plt.show()
st.pyplot(fig1)

st.subheader('  Price vs MA 44 vs MA100')
ma_100days = data.Close.rolling(100).mean()

fig2=plt.figure(figsize=(8,6))
plt.plot(ma_44days,'r' , label ='MA44')
plt.plot(ma_100days,'b' , label ='MA100')
plt.plot(data.Close , 'g' , label ='Closing Price')
plt.legend()
plt.show()
st.pyplot(fig2)

x=[]
y=[]

for i in range(44, data_test_scale.shape[0]):
  x.append(data_test_scale[i-44:i])
  y.append(data_test_scale[i,0])

x=np.array(x)
y=np.array(y)

predict = model.predict(x)

scale = scaler.scale_

predict = predict * scale
y=y*scale

st.subheader('Predicted vs Orignal Price')

fig3=plt.figure(figsize = (10,8))
plt.plot(predict , 'r', label='predicted price')
plt.plot(y , 'g' , label = 'Orignal Price')
xlabel = 'Time'
ylabel = 'Price'
plt.legend()
plt.show()
st.pyplot(fig3)

