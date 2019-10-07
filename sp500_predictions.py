from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from datetime import datetime
import pandas as pd
import numpy as np

#Predicting the S&P 500 

sphist = pd.read_csv('sphist.csv')
#print(sphist.head())


test = sphist.iloc[100:150].copy()

#convert Date to a Pandas date type
sphist['Date'] = pd.to_datetime(sphist.Date)

#sort dataframe by Date in ascending order
sphist = sphist.sort_values(by='Date', ascending=True)
#reindex after sorting
sphist = sphist.reset_index(drop=True)

#average price for the past 5 days
#average price for the past 30 days
#SD of price over the past 5 days

#for index, row in sphist.iterrows(): 
#    sphist['day_5'] = sphist['Close'].iloc[-5:index].mean()

#use rolling window function to calculate average Close price of previous 5 rows; use shift function here as well
sphist['day_5'] = sphist['Close'].rolling(5).mean()
sphist['day_5'] = sphist['day_5'].shift()

#use rolling window function to calculate average Close price of previous 30 days; use shift function here as well
sphist['day_30'] = sphist['Close'].rolling(30).mean()
sphist['day_30'] = sphist['day_30'].shift()

#use rolling window function to calculate SD pf Close price of previous 5 days; use shift function here as well
sphist['sd_5'] = sphist['Close'].rolling(5).std()
sphist['sd_5'] = sphist['sd_5'].shift()


#filter dataframe to only have rows beginning 1951-01-03 or greater
sphist = sphist[sphist["Date"] > datetime(year=1951, month=1, day=2)].copy()
#drop null rows
sphist = sphist.dropna(axis=0).copy()

train = sphist[sphist["Date"] < datetime(year=2013, month=1, day=1)].copy()
test = sphist[sphist["Date"] >= datetime(year=2013, month=1, day=1)].copy()

#initialize an instance of the LinearRegression class
lr = LinearRegression()

features = ['day_5','day_30','sd_5']
target = 'Close'

#fit model
lr.fit(train[features], train[target])
#predict based on model 
predictions = lr.predict(train[features])

#calculate root mean squared error for target and its predictions
train_rmse = np.sqrt(mean_squared_error(train[target], predictions))
print("Train RMSE: " + str(train_rmse))

