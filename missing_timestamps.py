import pandas as pd
from datetime import timedelta
import numpy as np


def remove_duplicates(data):
    """
    A function that removes duplicates in timestamps and removes
    timezone information from the timestamps 
    """
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    time = data["timestamp"].dt.tz_localize(None)
    data["timestamp"] = time
    data_processed = data.drop_duplicates(subset="timestamp", keep='last')
    data_processed = data_processed.set_index("timestamp")
    data_processed = data_processed.sort_index()

    return data_processed


def missing_data(data_processed):
    """
    A function that finds the dates of the missing data and fills it with the previous non-missing timestamp
    for a better visualization to present train/test data with the missing data
    """
    timestamps_series = pd.Series(data_processed.index)
    # zdaj bom spremenila hours=1, minutes=0, sam da vidm kolk manjka ce je frekvenca 1 h
    # drugac je bilo hours=0, minutes=5

    diff_5 = timedelta(hours=1)
    diff = timestamps_series.diff()
    more_than_5 = np.where(diff > diff_5)[0]
    previous_timestamp = more_than_5 - 1
    gaps = list()
    # a list of lists, where the first element is the previous timestamp of the missing data (succesive timestamps differ more than 5 min)
    #  and the second element is the end date of the missing data
    for i, j in zip(data_processed.iloc[previous_timestamp].index, data_processed.iloc[more_than_5].index):
        gaps.append([i, j])
    missing_timestamps = list()
    # dates of missing data where the start date is not a missing data
    for i in gaps:
        missing_timestamps.append(pd.date_range(start=i[0], end=i[1], freq='5T'))
    missing_timestamps_one_by_one = list()
    for i in missing_timestamps:
    # exclude the non-missing timestamps, the last timestamp is also excluded, because that is the index of the timestamp
    # that differs from the previous timestamp more than 5 min, so that timestamp exists is not missing
        for j in i[1:len(i)-1]:
            missing_timestamps_one_by_one.append(j)
    values = data_processed.iloc[previous_timestamp].values
    values_list = list()
    for i, v in enumerate(values):
        values_list.append(v[0])
    # fill the missing timestamps with the values of their previous non-missing timestamps
    missing_values_filled_previous = list()
    for t, v in zip(missing_timestamps, values_list):
        missing_values_filled_previous.append([v] * len(t[1:len(t)-1]))
    missing_values_filled_previous_one_by_one = list()
    for i in missing_values_filled_previous:
        for j in i:
            missing_values_filled_previous_one_by_one.append(j)
    
    return missing_timestamps_one_by_one, missing_values_filled_previous_one_by_one
