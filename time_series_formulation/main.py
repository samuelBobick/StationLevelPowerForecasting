import pandas as pd
from knn import KNN
from ffnn import NeuralNet
from STL import STLARIMA
from last_week import LastWeek
from similar_day import SimilarDay

data = pd.read_csv("/Users/sam/Desktop/StationLevelPowerForecasting/time_series_formulation/data.csv")
train = data.loc[:71423]
test = data.loc[71424:]

model = KNN()
model.fit(train)
rmse, wrmse, forecast = model.predict(test)
print('KNN:', "RMSE:", rmse, "WRMSE", wrmse)

model = NeuralNet()
model.fit(train)
rmse, wrmse, forecast = model.predict(test)
print('Neural Net:', "RMSE:", rmse, "WRMSE", wrmse)

# model = STLARIMA()
# model.fit(train)
# rmse, wrmse, forecast = model.predict(test)
# print('STL with ARIMA:', "RMSE:", rmse, "WRMSE", wrmse)

model = LastWeek()
rmse, wrmse, forecast = model.predict(test)
print('Last Week:', "RMSE:", rmse, "WRMSE", wrmse) 


model = SimilarDay()
rmse, wrmse, forecast = model.predict(test)
print('Last Week:', "RMSE:", rmse, "WRMSE", wrmse) 
