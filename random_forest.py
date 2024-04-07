from data_preparation import Preparation
from missing_timestamps import remove_duplicates
import pandas as pd
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.model_selection import grid_search_forecaster
from sklearn.ensemble import RandomForestRegressor
from skforecast.utils import save_forecaster
from skforecast.utils import load_forecaster
import matplotlib.pyplot as plt
from skforecast.model_selection import backtesting_forecaster
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect
# marvin
data = Preparation(r'/home/ieftimska/operato-meteo-1/data/MAS_processed/ELES-MAS-5001.csv.gz', "AMBIENT_TEMPERATURE")
#data = Preparation(r'/home/iva/Desktop/operato-meteo-1/data/MAS_processed/ELES-MAS-5001.csv.gz', "AMBIENT_TEMPERATURE")
train, test = data.split()
train_processed = remove_duplicates(train)
train_processed_ = train_processed["AMBIENT_TEMPERATURE"].copy().squeeze()
#train_processed = train_processed.asfreq("5T").fillna(method="ffill")
test_processed = remove_duplicates(test)
#test_processed = test_processed.asfreq("5T").fillna(method="ffill")
#test_processed_ = test_processed["AMBIENT_TEMPERATURE"].copy().squeeze()
whole_data = pd.concat([train_processed, test_processed])
whole_data = whole_data.rename(columns={"AMBIENT_TEMPERATURE": "y"})
whole_data.index = whole_data.index.rename("datetime")
whole_data_ = whole_data["y"].copy().squeeze()

'''forecaster = ForecasterAutoreg(regressor=RandomForestRegressor(random_state=123, n_jobs=-1, max_depth=10, n_estimators=100), lags=865)
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [5, 10, 15]
}
X_eval, y_eval = forecaster.create_train_X_y(y=whole_data_)'''
'''# Lags used as predictors
lags_grid = [i for i in range(1, 865)]

results_grid = grid_search_forecaster(
                   forecaster=forecaster,
                   y=whole_data_,
                   param_grid=param_grid,
                   lags_grid=lags_grid,
                   steps=864,
                   refit=False,
                   metric='mean_squared_error',
                   initial_train_size=len(train_processed_),
                   fixed_train_size=False,
                   return_best=True,
                   n_jobs='auto',
                   verbose=False,
                   show_progress=True
               )
print(results_grid)
#results_grid.to_csv("results_grid_search.csv")'''

#forecaster = ForecasterAutoreg(regressor=RandomForestRegressor(random_state=123, n_jobs=-1, max_depth=10, n_estimators=100), lags=865)
#forecaster.fit(y=whole_data.loc[:"2022", "y"])
#save_forecaster(forecaster, file_name='forecaster_random_forest_lags865.py', verbose=False)
# regressive method for multi step forecasting
#forecaster_loaded = load_forecaster('forecaster_random_forest_lags865.py', verbose=True)
#papa = whole_data.loc[:"2022"]
#print(papa[-864:])
#predictions = forecaster_loaded.predict(steps=864, last_window=papa[-865:].copy().squeeze())
#print(predictions)
'''metric, predictions = backtesting_forecaster(
                          forecaster=forecaster_loaded,
                          y=whole_data_,
                          initial_train_size=len(whole_data_.loc[:"2022"]),
                          steps=864,
                          refit=False,
                          fixed_train_size=False,
                          metric='mean_absolute_error',
                          verbose=True,
                          show_progress=False)
print(predictions)'''
#print(whole_data_["2023":])
#plt.plot(predictions)
#plt.show()
forecaster_direct = ForecasterAutoregDirect(regressor=RandomForestRegressor(random_state=123, n_jobs=-1, max_depth=10, n_estimators=100), lags=864, steps=864)
forecaster_direct.fit(y=train_processed_)
save_forecaster(forecaster_direct, file_name='direct_random_forest_lags864.py', verbose=False)
