import pandas as pd
from datetime import timedelta

parameter_type_list = ["AMBIENT_TEMPERATURE", "WIND_DIRECTION", "RELATIVE_HUMIDITY", "AIR_PRESSURE", "WIND_SPEED", "RAIN_INTENSITY", "SOLAR_RADIATION_INTENSITY"]


def another_format(path, parameter_type):
    """
    Converts the original csv file to a pandas series containing two columns
    timestamps and the chosen parameter type
    """
    document = pd.read_csv(path)
    one_parameter = document[document["parameter_type"] == parameter_type]
    time = pd.to_datetime(one_parameter['acquisition_timestamp'])
    parameter = one_parameter["value"]
    df_new = pd.DataFrame()
    df_new[f"{parameter_type}"] = parameter
    df_new["timestamp"] = time
    
    return df_new


def missing_data(data):
    """
    Adds additional column to the original csv file that indicates the time difference between 
    successive timestamps. In this way you check for a missing data
    """
    
    # reset the indices to be in an ascending order because the preprocessed csv file does not have succesive indices
    # this is needed for finding the previous and next element of the current element that has missing data
    dataset = data.reset_index()
    timestamps = dataset["timestamp"]
    # convert to pandas series
    timestamps_series = pd.Series(timestamps)
    diff_5 = timedelta(hours=0, minutes=5)
    dataset['Diff'] = timestamps_series.diff()
    dataset['More than 5 min'] = timestamps_series.diff() > diff_5
    index_more_5 = dataset[dataset["More than 5 min"] == True].index
    max_diff = dataset['Diff'].max()
    min_diff = dataset['Diff'].min()
    max_diff_idx = dataset['Diff'].idxmax()
    min_diff_idx = dataset['Diff'].idxmin()
    string = "Minimum difference between successive timestamps"
    dataset['Less than 5 min'] = timestamps_series.diff() < diff_5
    index_less_5 = dataset[dataset["Less than 5 min"] == True].index
    with open("missing_processed_data_station1.txt", 'w') as f:
        print(f"Minimum difference between successive timestamps {min_diff}\n", file=f)
        print(f"Maximum difference between successive timestamps {max_diff}\n", file=f)
        print(f"Number of less than 5 min succesive timestamps: {len(index_less_5)}.",file=f)
        print(f"Number of more than 5 min succesive timestamps: {len(index_more_5)}.",file=f)
        print('-' * len(string), file=f)
        print("Minimum difference: ", file=f) 
        print(f"{dataset.loc[min_diff_idx-1:min_diff_idx+1]}", file=f)
        for i in index_more_5:
            print(dataset.loc[i-1:i+1], file=f)
        print("Maximum difference: ", file=f)
        print(f"{dataset.loc[max_diff_idx-1:max_diff_idx+1]}", file=f)
        print('-' * len(string), file=f)
        print("Less than 5 min", file=f)
        for m in index_less_5:
            print(dataset.loc[m-1:m+1], file=f)
        
        return dataset

