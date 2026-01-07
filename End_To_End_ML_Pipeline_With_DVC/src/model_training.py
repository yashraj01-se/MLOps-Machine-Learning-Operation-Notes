import pandas as pd
import numpy as np
import os
import logging
from sklearn.ensemble import RandomForestClassifier
import pickle

########## Ensure the log directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# logger object
logger=logging.getLogger('model_training')
logger.setLevel('DEBUG')

# Console Handler object
console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

# File Handler object
file_handler=logging.FileHandler(os.path.join(log_dir,'model_training.log'))
file_handler.setLevel('DEBUG')

# Formatter
formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Adding handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(file_path:str) -> pd.DataFrame:
    """
    Load training and testing data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.
    """
    try:
        df= pd.read_csv(file_path)
        logger.debug(f"Data loaded successfully from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Parsing error while loading data from {file_path}: {e}")
        raise
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise   

def train_model(X_train: np.ndarray, y_train: np.ndarray, params:dict) -> RandomForestClassifier:
    """
    Docstring for train_model
    
    :param X_train: Description
    :type X_train: np.ndarray
    :param y_train: Description
    :type y_train: np.ndarray
    :param params: Description
    :type params: dict
    :return: Description
    :rtype: RandomForestClassifier
    """

    try:
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("Number of samples in X_train and y_train do not match.")
        
        logger.debug("Starting model training")
        clf=RandomForestClassifier(n_estimators=params.get('n_estimators',100),
                                   random_state=params.get('random_state',42))
        logger.debug(f"Model parameters: {params}")
        clf.fit(X_train,y_train)
        logger.debug("Model training completed successfully")
        return clf
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise
    except ValueError as e:
        logger.error(f"Value error: {e}")
        raise

def save_model(model: RandomForestClassifier, file_path: str)->None:
    """
    Docstring for save_model
    
    :param model: Description
    :type model: RandomForestClassifier
    :param file_path: Description
    :type file_path: str
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path,'wb') as f:
            pickle.dump(model,f)    
        logger.debug(f"Model saved successfully at {file_path}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise   

def main():
    try:
        params={'n_estimators':25,'random_state':2}
        train_data=load_data('data/processed/train_vectorized.csv')
        X_train=train_data.iloc[:,:-1].values
        y_train=train_data.iloc[:,-1].values

        clf=train_model(X_train,y_train,params)
        logger.debug("Model training process completed")

        model_save_path=os.path.join('models','random_forest_model.pkl')
        save_model(clf,model_save_path) 
        logger.debug(f"Model saved at {model_save_path}")
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise

if __name__=='__main__':
    main()