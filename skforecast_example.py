from sklearn.ensemble import RandomForestRegressor
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from missing_timestamps import remove_duplicates
from data_preparation import Preparation
from gluonts.time_feature import time_features_from_frequency_str, get_lags_for_frequency, get_seasonality
#data = Preparation(r'/home/iva/Desktop/operato-meteo-1/data/MAS_processed/ELES-MAS-5001.csv.gz', "AMBIENT_TEMPERATURE")
# marvin
data = Preparation(r'/home/ieftimska/operato-meteo-1/data/MAS_processed/ELES-MAS-5001.csv.gz', "AMBIENT_TEMPERATURE")
train, test = data.split()
train_processed = remove_duplicates(train)
#train_processed_ = train_processed["AMBIENT_TEMPERATURE"].copy().squeeze()
train_processed_ = train_processed.asfreq("5T").fillna(method="ffill")
test_processed = remove_duplicates(test)
test_processed_ = test_processed["AMBIENT_TEMPERATURE"].copy().squeeze()
lags_from_frequency = get_lags_for_frequency("5min")
seasonality = get_seasonality("5min")
forecaster = ForecasterAutoreg(regressor=RandomForestRegressor(random_state=123), lags=lags_from_frequency, differentiation=seasonality)
forecaster.fit(y=train_processed["AMBIENT_TEMPERATURE"])
predictions = forecaster.predict(steps=864)
predictions.to_csv("predictions_auto_features_skforecast.csv", index=False)