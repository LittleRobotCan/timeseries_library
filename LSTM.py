import numpy
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import io
import requests

from plot_tools import timeseries_plot

# normalize and split data into train vs test
def norm_split(dataset, split=0.67):
	scaler = MinMaxScaler(feature_range=(0, 1))
	dataset = scaler.fit_transform(dataset)
	# split into train and test sets
	train_size = int(len(dataset) * float(split))
	train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
	return train, test, scaler

# convert an array of values into a dataset matrix
# sliding window to create x and y
# x = i:i+lookback
# y = i+lookback
# for data = [1,2,3,4,5,6,7], if lookback = 3, x=1,2,3, y=4
# using the previous history of 1 to n datapoints to predict the next one
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return numpy.array(dataX), numpy.array(dataY)

def fit(x, y, look_back=1):
	# create and fit the LSTM network
	model = Sequential()
	model.add(LSTM(4, input_shape=(1, look_back)))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	model.fit(x, y, epochs=100, batch_size=1, verbose=2)
	return model


if __name__ == '__main__':
	# fix random seed for reproducibility
	numpy.random.seed(7)

	# load the dataset
	url = "https://app.quotemedia.com/quotetools/getHistoryDownload.csv?&webmasterId=501&startDay=02&startMonth=02&startYear=2002&endDay=02&endMonth=07&endYear=2009&isRanged=false&symbol=DOW"
	s=requests.get(url).content
	dataframe = pandas.read_csv(io.StringIO(s.decode('utf-8')))
	dataframe['date'] = pandas.to_datetime(dataframe['date'])
	dataframe = dataframe.sort_values(by='date')
	dataset = dataframe[['close']].values
	dataset = dataset.astype('float32')
	# reshape into X=t and Y=t+1
	train, test, scaler = norm_split(dataset)
	trainX, trainY = create_dataset(train)
	testX, testY = create_dataset(test)
	# reshape input to be [samples, time steps, features]
	trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
	testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
	# fit the model
	model = fit(trainX, trainY)