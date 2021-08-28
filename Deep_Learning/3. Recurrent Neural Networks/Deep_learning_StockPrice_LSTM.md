# Deep Learning and Stock Prices Using LSTM



#### Time Series Analysis

* Quite complex models
  * Combined **ARIMA** and **GARCH** model is working fine
  * Random walk model, Autoregressive model, Moving Average model → ARIMA
  * ARIMA + Generalised Autoagressive Heteroskedastic Model (GARCH)
* It can deal with volarility clustering : works fine during crisis



#### LSTM Architecture

* Black-box model, no underlying parameters to learn
  * Just learn the relationship between the features. (past stock prices)
* Works fine in the main : Not during crisis



### Training

* Data construction
  * 40 data points (features) and the price tomorrow (target variable)
* Training
  * Use LSTM to detect the relationship between the 40 features so between the prices in the past
* Prediction
  * 40 features so the prices in the past and predict the price tmr.

















































