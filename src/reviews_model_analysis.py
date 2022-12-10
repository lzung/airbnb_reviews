# author: Lauren Zung
# date: 2022-12-10

"""
Takes the preprocessed training and test data and tests multiple regressors.
Finds LGBMRegressor to be the best model.
 
Usage: src/reviews_model_selection.py --input_path=data/processed/ --output_path=results/model_selection/

Options:
--input_path=<input_path>           Path to the training data (in data/preprocessed)
--output_path=<output_path>         Path to the output directory for the tables and plots (results/model_selection/)

Command to run the script:
python src/reviews_model_selection.py --input_path=data/processed/ --output_path=results/model_selection/
"""
from docopt import docopt
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer, MinMaxScaler, QuantileTransformer
from sklearn.feature_extraction.text import CountVectorizer

opt = docopt(__doc__)

def main(input_path, output_path):
    """
    Takes the preprocessed training and test data and performs model training
    (using cross validation) with ridge, lasso, decision tree regressor and light GBM regressor
    Saves all results to the output path.

    Parameters
    ----------
    input_path : string
        The input directory that contains the preprocessed training data
    output_path : string
        The name of the directory that will contain the model selection plots and tables
    """

# Call main
if __name__ == "__main__":
    main(opt["--input_path"], opt["--output_path"])