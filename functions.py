import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# TODO: maybe with class

test_processed_compare = lambda test, end_time: test[test.index <= end_time]


def data_interval(data, start_time, end_time):
    intervals = data[(data['timestamp'] > start_time) & (data['timestamp'] < end_time)]
    
    return intervals


def plot_compare(predictions, alg, test_real):
    plt.figure(figsize=(20, 3))
    plt.title(f"Prediction for 72 hours, station 5001 MAS, {alg}")
    plt.plot(predictions.index, predictions[alg], label="predicted")
    plt.plot(test_real.index, test_real.values, label="real")
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Temperature[°C]")
    #plt.ylabel("Wind Direction")


def predict(path, test_real):
    predictions = pd.read_csv(path)
    predictions = predictions[predictions["ds"] >= '2023-01-01 00:20:00']
    predictions["ds"] = pd.to_datetime(predictions["ds"])
    test_compare = test_processed_compare(test_real, '2023-01-03 23:55:00')
    test_processed_regular = test_compare.asfreq("5T")
    ind = np.where(test_processed_regular.isna())[0]
    timestamps = test_processed_regular.iloc[ind].index
    predictions_ds = predictions.set_index("ds")
    predictions_ds = predictions_ds.drop(predictions_ds.loc[timestamps.values].index)

    return predictions_ds


def pdf(test_processed_compare, predictions, alg):
    error = test_processed_compare["AMBIENT_TEMPERATURE"].values - predictions[alg].values
    plt.hist(error, bins='auto', density=True)
    plt.title(f"PDF of error {alg} 72h forecast")
    mean_error = error.mean()
    ax = sns.kdeplot(data=error)
    kdeline = ax.lines[0]
    xs = kdeline.get_xdata()
    ys = kdeline.get_ydata()
    height = np.interp(mean_error, xs, ys)
    ax.vlines(mean_error, 0, height, color='black', ls=':')
    plt.show()