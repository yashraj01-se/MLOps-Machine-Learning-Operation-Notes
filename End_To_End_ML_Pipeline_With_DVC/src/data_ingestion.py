import pandas as pd
import os
from sklearn.model_selection import train_test_split
import logging
import yaml

########## Ensure the log directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

########### logging configuration
# Logger object
logger=logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

# Console Handler object
console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

# File Handler object
file_handler=logging.FileHandler(os.path.join(log_dir,'data_ingestion.log'))
file_handler.setLevel('DEBUG')

# Formatter
formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Adding handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_params(param_path: str) -> dict:
    """
    Load parameters from a YAML file.

    Args:
        param_path (str): Path to the YAML file.
    """
    try:
        with open(param_path, 'r') as f:
            params = yaml.safe_load(f)
        logger.debug(f"Parameters loaded successfully from {param_path}")
        return params
    except FileNotFoundError as e:
        logger.error(f"Parameter file not found: {e}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML file {param_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading parameters from {param_path}: {e}")
        raise

def load_data(data_url: str) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Args:
        data_url (str): URL to the CSV file."""
    try:
        df=pd.read_csv(data_url)
        logger.debug(f"Data loaded successfully from {data_url} with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {data_url}: {e}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Parsing error while loading data from {data_url}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise

def preprocess_data(df: pd.DataFrame)->pd.DataFrame:
    """
    Preprocess the data by separating features and target.

    Args:
        df (pd.DataFrame): The input DataFrame.
        feature_cols (list): List of feature column names.
        target_col (str): Target column name."""
    try:
        df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)
        df.rename(columns={'v1':'target','v2':'text'},inplace=True)
        logger.debug(f"Data preprocessing completed. Columns now: {df.columns.tolist()}")
        return df
    except KeyError as e:
        logger.error(f"Missing expected columns in DataFrame: {e}")
        raise
    except Exception as e:
        logger.error(f"Error during data preprocessing: {e}")
        raise

def save_data(train_df: pd.DataFrame, test_df: pd.DataFrame, data_path: str):   
    """
    Save the training and testing DataFrames to CSV files.

    Args:
        train_df (pd.DataFrame): Training DataFrame.
        test_df (pd.DataFrame): Testing DataFrame.
        train_path (str): Path to save the training CSV.
        test_path (str): Path to save the testing CSV."""
    try:
        raw_data_path=os.path.join(data_path,'raw')
        os.makedirs(raw_data_path,exist_ok=True)
        train_df.to_csv(os.path.join(raw_data_path,'train.csv'),index=False)
        test_df.to_csv(os.path.join(raw_data_path,'test.csv'),index=False)
        logger.debug(f"Training and testing data saved successfully at {raw_data_path}")    
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise   

def main():
    try:
        params=load_params('params.yaml')
        test_size=params['data_ingestion']['test_size']
        data_path='https://raw.githubusercontent.com/Vikashishere/Datasets/main/spam.csv'
        df=load_data(data_url=data_path)
        final_df=preprocess_data(df)
        train_df,test_df=train_test_split(final_df,test_size=test_size,random_state=42)
        save_data(train_df,test_df,data_path='./data')
    except Exception as e:
        logger.error(f"Error in data ingestion pipeline: {e}")
        raise

if __name__ == "__main__":
    main()