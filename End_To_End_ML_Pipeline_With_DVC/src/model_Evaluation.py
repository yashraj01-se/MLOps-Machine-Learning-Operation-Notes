import pandas as pd
import os
import logging
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import json

########## Ensure the log directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# logger object
logger=logging.getLogger('model_evaluation')
logger.setLevel('DEBUG')

# Console Handler object
console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

# File Handler object
file_handler=logging.FileHandler(os.path.join(log_dir,'model_evaluation.log'))
file_handler.setLevel('DEBUG')

# Formatter
formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Adding handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_model(model_path: str):
    """
    Load a trained model from a file.

    Args:
        model_path (str): Path to the model file.
    """
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logger.debug(f"Model loaded successfully from {model_path}")
        return model
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading model from {model_path}: {e}")
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.
    """
    try:
        df = pd.read_csv(file_path)
        logger.debug(f"Data loaded successfully from {file_path} with shape {df.shape}")
        return df
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Parsing error while loading data from {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise   

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Evaluate the model on the test data and return evaluation metrics.

    Args:
        model: The trained model.
        X_test (np.ndarray): Test features.
        y_test (np.ndarray): True labels for the test set.
    """
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc_auc
        }

        logger.debug(f"Model evaluation metrics: {json.dumps(metrics, indent=2)}")
        return metrics
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        raise
def save_metrics(metrics: dict, file_path: str) -> None:
    """
    Save evaluation metrics to a JSON file.

    Args:
        metrics (dict): Evaluation metrics.
        file_path (str): Path to save the metrics JSON file.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.debug(f"Evaluation metrics saved successfully at {file_path}")
    except Exception as e:
        logger.error(f"Error saving evaluation metrics to {file_path}: {e}")
        raise

def main():
    try:
        model_path = './models/random_forest_model.pkl'
        test_data_path = './data/processed/test_vectorized.csv'
        metrics_output_path = './reports/metrics/model_evaluation_metrics.json'

        # Load the trained model
        model = load_model(model_path)

        # Load the test data
        test_data = load_data(test_data_path)
        X_test = test_data.iloc[:, :-1].values
        y_test = test_data.iloc[:, -1].values

        # Evaluate the model
        metrics = evaluate_model(model, X_test, y_test)

        # Save the evaluation metrics
        save_metrics(metrics, metrics_output_path)
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise

if __name__ == '__main__':
    main()