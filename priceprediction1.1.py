from time import time
from math import sqrt
from numpy import concatenate
import numpy as np
import pandas as pd
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt


##############################################################################
# code from https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
##############################################################################


# changes from previous: Removing variable copies

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1) #### a bit worse without this 2% MAPE
	#for i in range(n_in, 0, -1):
	#	cols.append(df.shift(i))
	#	names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

def mean_absolute_percentage_error(y_true, y_pred): # create MAPE function
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# load dataset
dataset = read_csv('../Espana_new.csv', header=0, index_col=0,na_values=0,keep_default_na=True) #### adjusting NaN values
#Req sub,Reqs baj,Asi sub,Asi baj,Pre req,Enr abs,Enr net,Enr sub,Enr baj,Pre sub,Pre baj,Gen otr,Gen sol,Gen eol,Demanda
# use Espana.csv for the old case

sizelimit = 0#10000 # used to limit the dataset to recent years
dataset.dropna(inplace=True)
dataset = dataset.drop(dataset.index[0:sizelimit],axis=0)

price24 = dataset['Pre req'].values
price24 = price24[0:-24]
dataset = dataset.drop(dataset.index[0:24],axis=0)

dataset.index = pd.to_datetime(dataset.index, utc=True)
dataset['month'] = dataset.index.month
#weekday = dataset.index.weekday

weekday = np.array(dataset.index.weekday)
weekday[weekday < 5] = 0
weekday[weekday > 4] = 1
# perhaps make friday a different day 

dataset['weekday'] = weekday
dataset['hour'] = dataset.index.hour # check type
dataset['price-24'] = price24


print(weekday)

#print(dataset.head(48))     #### make sure to fix the NaN values, DONE
print(len(dataset))
print(dataset.shape)
values = dataset.values
#values = series_to_supervised(values, 1, 1,False) ## make false to include whole dataset


# integer encode direction
#encoder = LabelEncoder() ####### investigate
#values[:,4] = encoder.fit_transform(values[:,4])
# ensure all data is float
values = values.astype('float32')
#values = values.values

print("Values are:")
print(values)
print(values.shape)
#print(values.type)
print("\n")

X = np.delete(values,2,1) # price is 4 in old case
y = values[:, 2]

# use to merge assigned power into one column
#X[:,0] = X[:,0] + X[:,1]

print("The full X is: ")
print(X)
print(X.shape)
print("The full y is: ")
print(y)
print(y.shape)



# choosing variables
# old case
# X = X[:,[2,3,11,12,13,15,16,17]] # 
# new case
X = X[:,[0,1,2,3,4,5,6,7,8]] # use to change variables
# remeber to add activated power



# 4 % better to add assigned power 


#print("Used Columns: ")
#print(list(dataset)[2])#,3,12,13,14,17,18])
#print(list(dataset)[3])
#print(list(dataset)[12])
#print(list(dataset)[13])
#print(list(dataset)[14])
#print(list(dataset)[18])
#print(list(dataset)[19])
#data[:, [1, 9]]



y = y.reshape(-1, 1)

# normalize features
scalerX = MinMaxScaler(feature_range=(0, 1)).fit(X) # be sure of how this works, should have 1 as a standard deviation
scalery = MinMaxScaler(feature_range=(0, 1)).fit(y)
scaledX = scalerX.transform(X)
scaledy = scalery.transform(y)

# frame as supervised learning
X = series_to_supervised(scaledX, 1, 1,False) ## make false to include whole dataset
#y = series_to_supervised(y, 1, 1,True)
print(X) 

data = X
data['price'] = scaledy
print('\n')
#print(data)
#data = data.dropna() #inplace=True)

print("The length of X is: ", len(X))
print("The shape of X is: ", X.shape)
#print(y.shape())

##########################################################################
n_test_hours = 24 #* 50
n_train_hours = len(X) - n_test_hours
##########################################################################

# split into train and test sets, and input output
train_X = data.drop('price',axis=1).drop(data.index[n_train_hours:],axis=0).values # [:n_train_hours, :-1]
test_X = data.drop('price',axis=1).drop(data.index[:n_train_hours],axis=0).values # [n_train_hours:, :-1]

train_y = data['price'].drop(data.index[n_train_hours:],axis=0).values # [:n_train_hours, :-1]
test_y = data['price'].drop(data.index[:n_train_hours],axis=0).values

print(train_y.shape)
print(test_y.shape)


# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)


# (12228, 1, 10) (12228,) (24, 1, 10) (24,)



# design network
epoch = 25 # 15 better than 5
model = Sequential() # fix seed, use different seeds as bootstrap
model.add(LSTM(200, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(LSTM(200, return_sequences=True))
model.add(LSTM(200, return_sequences=True))
#model.add(LSTM(100, return_sequences=True))
#model.add(LSTM(100, return_sequences=True))
#model.add(LSTM(100, return_sequences=True))
model.add(LSTM(200))#, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

# change layers.


tensorboard = TensorBoard(log_dir="logs/{}".format(time()))

# fit network
history = model.fit(train_X, train_y, epochs=epoch, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# make a prediction
yhat = model.predict(test_X)

test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
test_y = test_y.reshape((len(test_y), 1))
#print('The prediction is: ', yhat)
#print('The value is :', test_X[:, 1:])
#print('The shape is :', test_X.shape)

inverse_ypredict = scalery.inverse_transform(yhat)
inverse_ytest = scalery.inverse_transform(test_y)
#print('Inv_y is: ', inverse_ypredict)

# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
#print('Inv_Yhat is: ', inv_yhat)
#print('Inv_Yhat shape is: ', inv_yhat.shape)

# invert scaling for actual
inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
#print('Inv_Y is: ', inv_y)
#print('Inv_Y shape is: ', inv_y.shape)


print('\n')
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE normalized concatenated matrix: %.3f' % rmse)

rmse = sqrt(mean_squared_error(inverse_ytest, inverse_ypredict))
print('Test RMSE non-normalized values: %.3f' % rmse)

rmse = sqrt(mean_squared_error(test_y, yhat))
print('Test RMSE normalized: %.3f' % rmse)
# calculate MAPE
mape = mean_absolute_percentage_error(inverse_ytest, inverse_ypredict)
print("Test MAPE non-normalized values: ", mape)

### plotting normalized forecast ###
#plt.title("Forecast vs Actual",fontsize=14) # dont scale the prediction
#plt.plot(yhat[:150], "b", linewidth=0.2, label="Forecast")
#plt.plot(test_y[:150], "r", linewidth=0.2,label="Actual")
#plt.legend(loc="upper left")
#plt.xlabel("Time Periods")
#plt.show()

### plotting forecast ###
plt.title("Forecast vs Actual",fontsize=14) # dont scale the prediction
plt.plot(inverse_ypredict[:150], "b", linewidth=0.2, label="Forecast")
plt.plot(inverse_ytest[:150], "r", linewidth=0.2,label="Actual")
plt.legend(loc="upper left")
plt.xlabel("Time Periods")
plt.show()



##### plot difference in forecasted price #####

# check data, be sure of scaling function
# first scale parameters (not price), then divide in train and test, be sure of order
# be sure of variables, choose which ones could be relevant
# find results
# another meeting sometime after 5th January


# limit variables and understand the series-to-supervised function

####### testing #######
# 2 layers 200 neurons gives 29% mape 16% on 24h
# 4 layers 200 neurons gives 25% mape
# 2 layers 400 neurons gives 26% mape 15% on 24h
# all (basically) same RMSE

## for new input data, larger dataset: 
# RMSE much smaller (factor 3) # these are with many NaN-values
# 2 layers 200 neurons gives 33% mape 16% on 24h
# 4 layers 200 neurons gives 34% mape 17% on 24h
# 2 layers 400 neurons gives 35% mape 16% on 24h

## for new input data, smaller dataset: 
# RMSE much smaller (factor 3) # these are with many NaN-values
# 2 layers 200 neurons gives 37% mape % on 24h
# 4 layers 200 neurons gives 35% mape 
# 2 layers 400 neurons gives 35% mape % on 24h

############## TO DO LIST #######

# DONE try with forecasted solar, wind and demand 
# DONE increase dataset

# DONE fix the NaN values (halves the dataset, something wrong?)
# try to merge assigned and demand and see how it affects accuracy
# assigned is probably a formula from demand
    # merging increases MAPE by 8 percentage points

# add real demand and check if thats the reason for the big change

# increase epoch, neuron and layers
 
# set a seed for the neural network that works good 
# set a for loop system for running the nn 

