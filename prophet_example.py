import pandas as pd
from prophet import Prophet
from data_preparation import *
from missing_timestamps import remove_duplicates, missing_data
from functions import *
from datetime import datetime

#data = Preparation(r'/home/iva/Desktop/operato-meteo-1/data/MAS_processed/ELES-MAS-5001.csv.gz', "AMBIENT_TEMPERATURE")
# marvin 
data = Preparation(r'/home/ieftimska/operato-meteo-1/data/MAS_processed/ELES-MAS-5001.csv.gz', "AMBIENT_TEMPERATURE")
train, test = data.split()
train_processed = remove_duplicates(train)
missing_timestamps_train, missing_values_filled_previous_train = missing_data(train_processed)
test_processed = remove_duplicates(test)
missing_timestamps_test, missing_values_filled_previous_test = missing_data(test_processed)
train_prophet = pd.DataFrame()
train_prophet["ds"] = train_processed.index
train_prophet["y"] = train_processed["AMBIENT_TEMPERATURE"].values
test_prophet = pd.DataFrame()
test_prophet["ds"] = test_processed.index
start_time = datetime.now()
m = Prophet()
m.fit(train_prophet)
forecast = m.predict(test_prophet)
forecast.to_csv("prophet_forecasts.csv", index=False)
end_time = datetime.now()
print(f"Duration took {end_time-start_time}")

