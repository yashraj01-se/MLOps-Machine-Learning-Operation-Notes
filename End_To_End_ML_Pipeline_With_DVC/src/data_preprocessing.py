from pydoc import text
import pandas as pd
import os
import logging
import nltk
import string
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
nltk.download('stopwords') # Downloading stopwords from NLTK
nltk.download('punkt') # Downloading punkt tokenizer from NLTK
nltk.download('punkt_tab') # Downloading punkt_tab tokenizer from NLTK

########## Ensure the log directory exists
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# logger object
logger=logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

# Console Handler object
console_handler=logging.StreamHandler()
console_handler.setLevel('DEBUG')

# File Handler object
file_handler=logging.FileHandler(os.path.join(log_dir,'data_preprocessing.log'))
file_handler.setLevel('DEBUG')

# Formatter
formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Adding handlers to logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def transfrom_text(text: str) -> str:
    """
    Transform the input text by lowercasing, removing punctuation, tokenizing,
    removing stopwords, and stemming.

    Args:
        text (str): The input text to be transformed."""
    ps=PorterStemmer()
    text=text.lower() # Lowercasing the text
    text=nltk.word_tokenize(text) # Tokenizing the text
    text=[word for word in text if word.isalnum()] # Removing punctuation
    text=[word for word in text if word not in stopwords.words('english')] # Removing stopwords
    text=[ps.stem(word) for word in text] # Stemming the words
    return " ".join(text) # Joining the words back into a single string

def preprocess_data(df: pd.DataFrame, text_column='text', target_col='target') -> pd.DataFrame:
    """
    Preprocess the data by transforming text features and encoding the target variable.

    Args:
        df (pd.DataFrame): The input DataFrame.
        text_column (str): Text column name.
        target_col (str): Target column name."""
    try:
        # Transform text features
        logger.debug("Starting text transformation")
        encoder=LabelEncoder()
        df[target_col]=encoder.fit_transform(df[target_col]) # Encoding the target variable
        logger.debug(f"Target column '{target_col}' encoded successfully")

        df=df.drop_duplicates(keep='first') # Dropping duplicate rows
        logger.debug("Duplicate rows dropped successfully")

        #apply transform_text function to text column
        df[text_column]=df[text_column].apply(transfrom_text)
        logger.debug(f"Text column '{text_column}' transformed successfully")
        return df
    except KeyError as e:
        logger.error(f"Missing expected columns in DataFrame: {e}")
        raise
    except Exception as e:
        logger.error(f"Error during data preprocessing: {e}")
        raise

def main(text_column='text', target_col='target'):
    """
    Docstring for main
    
    :param text_column: Description
    :param target_col: Description
    """
    try:
        train_data=pd.read_csv('data/raw/train.csv')
        test_data=pd.read_csv('data/raw/test.csv')
        logger.debug("Raw data loaded successfully")

        train_prerocessed_data=preprocess_data(train_data,text_column,target_col)
        test_prerocessed_data=preprocess_data(test_data,text_column,target_col)
        logger.debug("Data preprocessing completed for both training and testing data")


        #Store the Preprocessed data
        data_path=os.path.join('./data','interim')
        os.makedirs(data_path,exist_ok=True)

        train_prerocessed_data.to_csv(os.path.join(data_path,'train_preprocessed.csv'),index=False)
        test_prerocessed_data.to_csv(os.path.join(data_path,'test_preprocessed.csv'),index=False)
        logger.debug(f"Preprocessed data saved successfully at {data_path}")

    except Exception as e:
        logger.error(f"Error in main function: {e}")
        raise
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except pd.errors.EmptyDataError as e:
        logger.error(f"Empty data error: {e}")
        raise   
    except Exception as e:
        logger.error(f"Transformation error: {e}")
        raise

if __name__=="__main__":
    main()




