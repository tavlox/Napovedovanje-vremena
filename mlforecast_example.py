from sklearn.ensemble import RandomForestRegressor
from data_preparation import Preparation
import pandas as pd
from mlforecast import MLForecast
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error
from missing_timestamps import remove_duplicates
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb
from mlforecast.target_transforms import Differences
from numba import njit
from window_ops.expanding import expanding_mean
from window_ops.rolling import rolling_mean
from gluonts.time_feature import time_features_from_frequency_str, get_lags_for_frequency, get_seasonality
from datetime import datetime
import pickle
import numpy as np
import pickle

# marvin
#data = Preparation(r'/home/ieftimska/operato-meteo-1/data/MAS_processed/ELES-MAS-5001.csv.gz', "WIND_DIRECTION")
data = Preparation(r'/home/ieftimska/operato-meteo-1/data/MAS_processed/ELES-MAS-5001.csv.gz', "AMBIENT_TEMPERATURE")
#data = Preparation(r'/home/iva/Desktop/operato-meteo-1/data/MAS_processed/ELES-MAS-5001.csv.gz', "AMBIENT_TEMPERATURE")
train, test = data.split()
train_processed = remove_duplicates(train)
#train_processed_ = train_processed["AMBIENT_TEMPERATURE"].copy().squeeze()
#train_processed_ = train_processed.asfreq("5T").fillna(method="ffill")
test_processed = remove_duplicates(test)
#test_processed_ = test_processed["AMBIENT_TEMPERATURE"].copy().squeeze()
train_processed_one_hour = train_processed.asfreq("1H")#.dropna()
test_processed_one_hour = test_processed.asfreq("1H").dropna()
idx = np.where(train_processed_one_hour.isnull())[0]
train_processed_one_hour.iloc[idx] = 50
int_indices = [t for t in range(1, len(train_processed_one_hour)+1)]
dataset_train = pd.DataFrame()
dataset_train["ds"] = train_processed_one_hour.index
dataset_train["y"] = train_processed_one_hour["AMBIENT_TEMPERATURE"].values
dataset_train["unique_id"] = "AMBIENT_TEMPERATURE"
#dataset_train["y"] = train_processed["WIND_DIRECTION"].values
#dataset_train["unique_id"] = "WIND_DIRECTION"



#models = [RandomForestRegressor(random_state=123)]
models = [xgb.XGBRegressor(random_state=123, missing=50)]
#models = [lgb.LGBMRegressor(random_state=123)]
'''model = MLForecast(models=models,
                    freq="5T",
                    lags=[t for t in range(1, 1440)],
                    date_features=['dayofweek', 'month'],
                    num_threads=6)'''
#lags_from_frequency = get_lags_for_frequency("5min")
#seasonality = get_seasonality("5min")
#papa = time_features_from_frequency_str("1H")
lags_from_frequency = get_lags_for_frequency("1H")
seasonality = get_seasonality("1H")

# za 5 min predikcija: date_features=['minute', 'hour', 'dayofweek', 'day', 'dayofyear'],


model = MLForecast(models=models,
                    freq="1H",
                    lags=lags_from_frequency,
                    date_features=['hour', 'dayofweek', 'day', 'dayofyear'],
                    target_transforms=[Differences([seasonality])],
                    num_threads=6)
start_time = datetime.now()

# direct multi-step strategy
'''model.fit(dataset_train, max_horizon=864)
predictions = model.predict(864)
end_time = datetime.now()
predictions.to_csv("predictions_auto_features_direct_strategy.csv", index=False)
duration = end_time - start_time
with open("duration.txt", 'wb') as d:
    d.write(duration)'''
model.fit(dataset_train)
# save the model to disk
#pickle.dump(model, open("forecast_1h.pkl", 'wb'))

# included whole? (idk if this is ok) train samples, to get prediction at 2023-01-01 00:20:00
# preeska bese train_processed_one_hour_compare[-745:]
train_processed_one_hour_compare = train_processed.asfreq("1H").dropna()
train_test = pd.concat([train_processed_one_hour_compare, test_processed_one_hour])



# iterating through test samples to get predictions for whole test data, 745 because the largerst lag is 745 and seasonality of lag 24
# preeska bese test_processed_one_hour kaj dataset_test i bese in range(len(test_processed_one_hour))
# 19059 ti e posleden train sample
predictions_all_iterations = pd.DataFrame()
for i in range(19059, len(train_test)):
    dataset_test = pd.DataFrame()
    dataset_test["ds"] = train_test.index[0:i]
    dataset_test["y"] = train_test["AMBIENT_TEMPERATURE"].values[0:i]
    dataset_test["unique_id"] = "AMBIENT_TEMPERATURE"
    predictions = model.predict(horizon=1, new_data=dataset_test)
    predictions_all_iterations = pd.concat([predictions_all_iterations, predictions])

predictions_all_iterations.to_csv("predictions_xgb_freq_1h_forecast_1h_removed_seasonality_first_samples_test.csv")
#prediction = model.predict(horizon=1)
#prediction.to_csv("first_prediction_forecast_1h_freq_1h.csv", index=False)

end_time = datetime.now()
print(f"Duration took {end_time-start_time}")



#prepared_df = model.preprocess(dataset_train, id_col='unique_id', time_col='ds', target_col='y', static_features=[])
#X_train, y_train = prepared_df.drop(columns=['unique_id', 'ds', 'y']), prepared_df['y']
#model.fit_models(X_train, y_train)
#predictions = model.predict(864)

#model.fit(dataset_train, fitted=True, max_horizon=864)
#train_predictions = model.forecast_fitted_values()
#train_predictions.to_csv("train_predictions_mlforecast_24lags.csv")
