from data_preparation import Preparation
import argparse
import pandas as pd
import logging
import datetime
from autogluon.timeseries import TimeSeriesPredictor
import sys
from tqdm import tqdm
from autogluon.tabular import TabularPredictor
from autogluon.core.dataset import TabularDataset
from autogluon.timeseries.models import RecursiveTabularModel

logger = logging.getLogger(__name__)
logging.basicConfig(filename='logging.log', encoding='utf-8', level=logging.NOTSET, format='%(asctime)s %(message)s')
logging.info("Running the main script.")

#data = Preparation(r'/home/iva/Desktop/operato-meteo-1/data/MAS_processed/ELES-MAS-5001.csv.gz', "AMBIENT_TEMPERATURE")
# remote na marvin 
data = Preparation(r'/home/ieftimska/operato-meteo-1/data/MAS_processed/ELES-MAS-5001.csv.gz', "AMBIENT_TEMPERATURE")
train, test = data.split()
train_prepared = data.fill_missing_data(train)
test_filled = data.fill_missing_data(test)
#train_data = TabularDataset(train)


def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('-mode', type=str, default='train', help='choose training or evaluation mode')
        parser.add_argument('-model_quality', type=str, default='medium_quality', help='choose model quality')
        parser.add_argument('-path', type=str, default='saved model', help='choose path to save the trained model')
        parser.add_argument('-prediction_length', type=int, default=1, help='choose the prediction length')
        
        args = parser.parse_args()
        
        if args.mode == "train":
            logging.info("Training the model.")
            time_start = datetime.datetime.now()
            #predictor = TimeSeriesPredictor(prediction_length=args.prediction_length, path=args.path, target="target", eval_metric="sMAPE", verbosity=3)
            #predictor.fit(train_prepared, presets=args.model_quality)
            predictor = TimeSeriesPredictor(prediction_length=args.prediction_length, path=args.path, target="target", eval_metric="sMAPE", verbosity=3)
            predictor.fit(train_prepared, presets=args.model_quality,  hyperparameters={"DirectTabular": {
                "tabular_hyperparameters": {
                    "GBM": {}, }, "subsampling_strategy": {"items"}}}, hyperparameter_tune_kwargs=None, enable_ensemble=False)
            time_end = datetime.datetime.now()
            duration = time_end - time_start
            msg_duration = f"The model was trained in {duration}."
            logging.info(msg_duration)

            return
       
        elif args.mode == "eval":
            print("Preparing the test data...")
            print("--------------------------")
            test_prepared = data.prepare_test(test_filled)
            print("Test data prepared, now evaluate the best trained model.")
            logging.info("Evaluating the best trained model.")
            # load the best trained model
            predictor = TimeSeriesPredictor.load(r'/home/ieftimska/operato-meteo-1/LightGBM_72h/', require_version_match=True)
            time_start = datetime.datetime.now()
            predictions = pd.concat([predictor.predict(k, use_cache=False) for k in tqdm(test_prepared[1:1500])]) 
            time_end = datetime.datetime.now()
            duration = time_end - time_start
            msg_duration = f"Predictions took {duration}."
            logging.info(msg_duration)          
            
            return predictions
            
    except BaseException as b:
        if sys.argv[1] != '-h':
            msg = f"{b}."
            logging.error(msg)
            raise Exception(msg)


if __name__ == '__main__':
    predictions = main()
    #predictions.to_csv("predictions_300_Recursive_Tabular_72h_LightGBM.csv")
   
