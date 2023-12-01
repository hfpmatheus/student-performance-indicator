import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from src.logger import logging

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            logging.info("Started prediction pipeline")
            preprocessor_path = 'artifacts/objects/preprocessor.pkl'
            model_path = 'artifacts/models/model.pkl'
            
            # Loading preprocessor and model objects
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)
            logging.info("Loaded preprocessor and model objects.")

            # Transforming data
            X = preprocessor.transform(features)
            logging.info("Preprocessed the data input array.")

            # Prediction
            y_pred = model.predict(X)
            logging.info("Predict complete.")
            return y_pred

        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self,
                 gender: str,
                 race_ethnicity,
                 parental_level_of_education,
                 lunch,
                 test_preparation_course):
        
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course]
            }

            return pd.DataFrame(custom_data_input_dict)

        except:
            pass