# author: Lauren Zung
# date: 2022-12-10

"""Cleans, splits and pre-processes the data from the New York City Airbnb listings (2019) dataset.
   Original source: https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data
   Writes the training and test data to separate csv files.
   
Usage: src/data_preprocess.py --input_path=<input_path> --output_path=<output_path>

Options:
--input_path=<input_path>           Path of the input directory that contains the raw data
--output_path=<output_path>         Path of the output directory which will contain the partitioned data 

Command to run the script:
python src/preprocess_data.py --input_path=data/raw/ --output_path=data/processed/
"""

from docopt import docopt
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

opt = docopt(__doc__)

sid = SentimentIntensityAnalyzer()

def main(input_path, output_path):
    """This function takes the raw data and generates train and test splits as data frames
    Parameters
    ----------
    input_path : string
        Path of the input directory that contains the raw data
    output_path : string
        Path of the output directory which will contain the partitioned data 
    """ 
    # Check if the data processed directory exists. If it doesn't, create the new folder
    try:
        isDirExist = os.path.isdir(output_path)
        if not isDirExist:
          print("Directory does not exist! Creating the path!")
          os.makedirs(output_path)
    
    except Exception as ex:
        print("Exception occurred :" + ex)
        exit()
        
    # Read the data from input
    nyc_data = pd.read_csv(f"{input_path}AB_NYC_2019.csv")

    # Remove examples that have a null number of reviews per month
    nyc_data = nyc_data[nyc_data['reviews_per_month'].notna()]

    # Change listing names that are null to empty string
    nyc_data['name'] = nyc_data['name'].fillna('')
    
    # Split data into train and test sets
    train_df, test_df = train_test_split(nyc_data, test_size=0.50, random_state=123)

    # Perform feature engineering on text feature
    train_df = train_df.assign(n_words=train_df["name"].apply(get_length_in_words))
    train_df = train_df.assign(vader_sentiment=train_df["name"].apply(get_sentiment))
    train_df = train_df.assign(rel_char_len=train_df["name"].apply(get_relative_length))

    test_df = test_df.assign(n_words=test_df["name"].apply(get_length_in_words))
    test_df = test_df.assign(vader_sentiment=test_df["name"].apply(get_sentiment))
    test_df = test_df.assign(rel_char_len=test_df["name"].apply(get_relative_length))
    
    # Transform outputs into csv file
    train_df.to_csv(f"{output_path}airbnb_train_df.csv")
    test_df.to_csv(f"{output_path}airbnb_test_df.csv")
    
    # Run tests to verify that the train_df and the test_df are correctly saved
    assert os.path.isfile(f"{output_path}airbnb_train_df.csv"), "Could not find the train_df in the data processed directory." 
    assert os.path.isfile(f"{output_path}airbnb_test_df.csv"), "Could not find the test_df in the data processed directory." 

def get_relative_length(title, AIRBNB_ALLOWED_CHARS=50.0):
    """
    Returns the relative length of text.

    Parameters:
    ------
    title: (str)
    the title of a listing

    Keyword arguments:
    ------
    AIRBNB_ALLOWED_CHARS: (float)
    the denominator for finding relative length;
    Airbnb allows up to 50 characters for a title (name)

    Returns:
    -------
    relative length of text: (float)

    """
    return len(title) / AIRBNB_ALLOWED_CHARS


def get_length_in_words(title):
    """
    Returns the length of the title in words.

    Parameters:
    ------
    title: (str)
    the title of a listing

    Returns:
    -------
    length of tokenized text: (int)

    """
    return len(nltk.word_tokenize(title))


def get_sentiment(title):
    """
    Returns the compound score representing the sentiment: -1 (most extreme negative) and +1 (most extreme positive)
    The compound score is a normalized score calculated by summing the valence scores of each word in the lexicon.

    Parameters:
    ------
    title: (str)
    the title of a listing

    Returns:
    -------
    sentiment of the text: (str)
    """
    scores = sid.polarity_scores(title)
    return scores["compound"]

if __name__ == "__main__":
    main(opt["--input_path"], opt["--output_path"])
    
    
