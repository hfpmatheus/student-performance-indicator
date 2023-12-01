import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
import pickle
from sklearn.metrics import mean_absolute_error
from src.logger import logging

def save_object(file_path, object):
    try:
        directory_path = os.path.dirname(file_path)
        os.makedirs(directory_path, exist_ok=True)

        with open(file_path, "wb") as file:
            pickle.dump(object, file)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        results = {}

        for model_name, model in models.items():
            model.fit(X_train, y_train)

            # Make the predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Evaluate predictions
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)

            # Print results
            print(model_name)
            print('Model performance for training set')
            print(f'- Mean Absolute error: {train_mae}')

            print('----------------------------------')

            print('Model performance for test set')
            print(f'- Mean Absolute error: {test_mae}')

            print('\n==================================\n')
            results[model_name] = test_mae

        # Get best model score, name and object
        best_model_score = max(sorted(results.values())) 
        best_model_name = list(results.keys())[
            list(results.values()).index(best_model_score)
        ]
        
        best_model_object = models[best_model_name].fit(X_train, y_train)

        if best_model_score < 0.6:
            raise CustomException("No best model found")
        logging.info("Obtained best model.")
            
        return results, best_model_object
    
    except Exception as e:
        raise CustomException(e, sys)