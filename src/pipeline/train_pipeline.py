from src.logger import logging
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainPipeline:
    def train(self):
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

        data_transformation = DataTransformation()
        train_arr, test_arr = data_transformation.start_data_transformation(train_data_path, test_data_path)

        model_trainer = ModelTrainer()
        models_report, model_object_file_path = model_trainer.initiate_model_trainer(train_arr, test_arr)
        return models_report