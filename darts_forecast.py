from darts.models import RandomForest
from missing_timestamps import remove_duplicates
from data_preparation import Preparation
from darts import TimeSeries
from darts.utils.statistics import plot_acf, check_seasonality
data = Preparation(r'/home/ieftimska/operato-meteo-1/data/MAS_processed/ELES-MAS-5001.csv.gz', "AMBIENT_TEMPERATURE")
train, test = data.split()
train_processed = remove_duplicates(train)
train_processed_ = train_processed["AMBIENT_TEMPERATURE"].copy().squeeze()
#train_processed_ = train_processed.asfreq("5T").fillna(method="ffill")
#test_processed = remove_duplicates(test)
#test_processed_ = test_processed["AMBIENT_TEMPERATURE"].copy().squeeze()
#train_series = TimeSeries.from_values(train_processed_.values)
#model = RandomForest(lags=864, n_estimators=100, max_depth=10, criterion="absolute_error", use_static_covariates=False, output_chunk_length=864, multi_models=False)
#model.fit(train_series)
#model.save("random_forest_lags864.pkl")
train_darts = TimeSeries.from_values(train_processed["AMBIENT_TEMPERATURE"].values)
# check seasonality in train
for m in range(2, 105121):
    is_seasonal, period = check_seasonality(train_darts, m=m, max_lag=105120, alpha=0.05)
    if is_seasonal:
        print("There is seasonality of order {}.".format(period))