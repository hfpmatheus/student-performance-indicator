import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.components.data_ingestion import DataIngestion


@dataclass
class DataTransformationConfig:
    preprocessor_object_file_path: str = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def preprocessor_object(self):
        '''
        This function creates de pipeline for preprocessing the features.
        '''

        try:
            categorical_columns = ["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course"]

            categorial_encoder_pipeline = Pipeline(
                steps=[
                    ("encoder", OneHotEncoder())
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("categorical_pipeline", categorial_encoder_pipeline, categorical_columns)
                ],
                sparse_threshold=0
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
    
    def start_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Finished reading train and test data")

            logging.info("Started preprocessing")
            # Dropping target feature
            target_column = "average_score"
            train_target_ft = train_df[target_column].values
            test_target_ft = test_df[target_column].values
            train_df = train_df.drop(columns=[target_column], axis=1)
            test_df = test_df.drop(columns=[target_column], axis=1)

            # Calling preprocessor object
            preprocessor_object = self.preprocessor_object()
            logging.info("Applying preprocessor object to train and test data")

            train_preprocessed = preprocessor_object.fit_transform(train_df)
            test_preprocessed = preprocessor_object.transform(test_df)

            train_arr = np.c_[
                train_preprocessed, train_target_ft
            ]

            test_arr = np.c_[
                test_preprocessed, test_target_ft
            ]

            logging.info("Preprocessing finished")
            
            save_object("artifacts/objects/preprocessor.pkl", preprocessor_object)
            logging.info("Saved preprocessor object in pickle file.")

            return (train_arr, test_arr, self.data_transformation_config.preprocessor_object_file_path)
        
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == '__main__':
    data_ingestion = DataIngestion()
    train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, preprocessor_object_file_path = data_transformation.start_data_transformation(train_data_path, test_data_path)