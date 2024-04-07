import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame
from checking_timestamps import another_format
import logging
from tqdm import tqdm


class Preparation:
    def __init__(self, path, parameter_type):
        self.path = path
        self.parameter_type = parameter_type
    
    def split(self):
        """
        Splits the original csv file in train and test data.
        2020, 2021, 2022 are used for train set, 2023 is used for test set.
        """
        logging.info("Split the dataset to train and test data.")
        try:
            document = another_format(self.path, self.parameter_type)
            # convert to string to use the function startswith
            document["timestamp"] = (document["timestamp"]).astype(str)
            three_years = document[(document["timestamp"].str.startswith('2020')) + (document["timestamp"].str.startswith('2021')) + (document["timestamp"].str.startswith('2022'))]
            one_year = document[(document["timestamp"].str.startswith('2023'))]
        except BaseException as b:
            msg = f"{b}."
            logging.error(msg)
            raise Exception(msg)

        return three_years, one_year

    def fill_missing_data(self, data_tr):
        """
        A function that removes duplicated values from the dataframe and fill the missing values
        to achieve a regular frequency of 5 min. It also prepares the data in a suitable format to use AUTOGLUON.
        """
        logging.info("Fill mising values and prepare the data in a suitable format to use AUTOGLUON.")
        try:
            data = data_tr.drop_duplicates(subset="timestamp", keep='last')
            data_prepared = pd.DataFrame()
            data["timestamp"] = pd.to_datetime(data["timestamp"])
            time = data["timestamp"].dt.tz_localize(None)
            data_prepared["timestamp"] = time
            data_prepared["target"] = data[f"{self.parameter_type}"]
            data_prepared["item_id"] = self.parameter_type
            data_prepared = TimeSeriesDataFrame.from_data_frame(data_prepared, id_column="item_id", timestamp_column="timestamp")
            data_regular = data_prepared.to_regular_index(freq="5T")
            df_filled = data_regular.fill_missing_values()
        
        except BaseException as b:
            msg = f"{b}."
            logging.error(msg)
            raise Exception(msg)
        
        return df_filled

    def prepare_test(self, data):
        logging.info("Prepare the test data.")
        """
        It returns test_data_list, a list of DataFrames where the last elements of each DataFrame differ for 5 min.
        E.g [(23:40, 23:45, 23:50), (23:45, 23:50, 23:55)...]. The test_data_list is then used in the predict function
        of AutoGluon to get the predicted outputs of the best trained model. The predict function predicts a value according
        to the last timestamp in the DataFrame and the chosen prediction length(in this case 1, predicting for the next 5 min), 
        so if the last timestamp in a DataFrame 23:40, the predict function returns predicted value at 23:45. 
        That is why the test_data_list needs to be given in that format in order to get predicted results 
        for different timstamps, not just a point prediction value.
        """
        freq_5_min = data.index.get_level_values('timestamp')
        parameter = data["target"].values
        try:
            test_data_list = []
            for i, v in enumerate(tqdm(freq_5_min[0:len(freq_5_min)-299])):
                # get a list of three values
                three_values = parameter[i:i+300]
                # take last three elements in every iteration
                three_values = three_values[-300:]
                freq = freq_5_min[i:i+300]
                freq = freq[-300:]
                df_test = pd.DataFrame()
                df_test["timestamp"] = freq
                df_test["target"] = three_values
                df_test["item_id"] = self.parameter_type
                test_data = TimeSeriesDataFrame.from_data_frame(df_test, id_column="item_id", timestamp_column="timestamp")
                test_data_list.append(test_data)
        except BaseException as b:
            msg = f"{b}."
            logging.error(msg)
            raise Exception(msg)
        
        return test_data_list


        
#data = Preparation(r'/home/iva/Desktop/operato-meteo-1/data/MAS_processed/ELES-MAS-5001.csv.gz', "AMBIENT_TEMPERATURE")
# remote na marvin 
#data = Preparation(r'/home/ieftimska/operato-meteo-1/data/MAS_processed/ELES-MAS-5001.csv.gz', "AMBIENT_TEMPERATURE")
#train, test = data.split()
#train_prepared = data.fill_missing_data(train)
#print(train_prepared)
#test_filled = data.fill_missing_data(test)
#test_data_list = data.prepare_test(test_filled)
#print(test_data_list[0])'''
