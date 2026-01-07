import pandas as pd
import os
import logging
from sklearn.feature_extraction.text import TfidfVectorizer 

########## Ensure the log directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# logger object
logger=logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

# Console Handler object
console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

# File Handler object
file_handler=logging.FileHandler(os.path.join(log_dir,'feature_engineering.log'))
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
    Load data from a CSV file.

    Args:
        file_path (str): Path to the CSV file.
    """
    try:
        df=pd.read_csv(file_path)
        df.fillna('',inplace=True) # Handling missing values by filling with empty strings
        logger.debug(f"Data loaded successfully from {file_path} with shape {df.shape}")
        return df
    except pd.errors.ParserError as e:
        logger.error(f"Parsing error while loading data from {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        raise
def apply_tfidf_vectorization(train_data:pd.DataFrame,test_data:pd.DataFrame,max_features:int) -> tuple:
    """
    Apply TF-IDF vectorization to the specified text column in the DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.
        text_column (str): The name of the text column to be vectorized.
    """
    try:
        tfidf_vectorizer=TfidfVectorizer(max_features=max_features)
        X_train=train_data['text'].values
        y_train=train_data['target'].values
        X_test=test_data['text'].values
        y_test=test_data['target'].values

        X_train_tfidf=tfidf_vectorizer.fit_transform(X_train)
        X_test_tfidf=tfidf_vectorizer.transform(X_test)

        train_df=pd.DataFrame(X_train_tfidf.toarray())
        train_df['target']=y_train

        test_df=pd.DataFrame(X_test_tfidf.toarray())
        test_df['target']=y_test

        logger.debug(f"TF-IDF vectorization applied successfully.")
        return train_df, test_df
    except Exception as e:
        logger.error(f"Error during TF-IDF vectorization: {e}")
        raise

def save_data(df: pd.DataFrame, file_path: str):   
    """
    Save the training and testing DataFrames to CSV files.

    Args:
        train_df (pd.DataFrame): Training DataFrame.
        test_df (pd.DataFrame): Testing DataFrame.
        train_path (str): Path to save the training CSV.
        test_path (str): Path to save the testing CSV.
    """
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        df.to_csv(file_path,index=False)
        logger.debug(f"Data saved successfully at {file_path}")
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise

def main():
    """
    Docstring for main
    """
    try:
        max_features=50
        train_data=load_data('data/interim/train_preprocessed.csv')
        test_data=load_data('data/interim/test_preprocessed.csv')

        train_vectorized_data,test_vectorized_data=apply_tfidf_vectorization(train_data,test_data,max_features)
        logger.debug("TF-IDF vectorization completed for both training and testing data")

        #Store the Vectorized data
        data_path=os.path.join('./data','processed')

        save_data(train_vectorized_data, os.path.join(data_path, 'train_vectorized.csv'))
        save_data(test_vectorized_data, os.path.join(data_path, 'test_vectorized.csv'))
        logger.debug(f"Vectorized data saved successfully at {data_path}")
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise 

if __name__=='__main__':
    main()