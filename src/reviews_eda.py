# author: Lauren Zung
# date: 2022-12-10

"""Creates new tables and plots of the training data from the NYC Airbnb (2019) dataset.
Saves the tables and plots as png files.

Original data source: https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data

Usage: src/reviews_eda.py --input_path=<input_path> --output_path=<output_path>

Options:
--input_path=<input_path>           Path to the training data (in data/preprocessed)
--output_path=<output_path>         Path to the output directory for the tables and plots (eda/)

Command to run the script:
python src/reviews_eda.py --input_path=data/processed/ --output_path=eda/
"""
from docopt import docopt
import os
import pandas as pd
# import altair_ally as aly
# import vl_convert as vlc
import matplotlib.pyplot as plt
# import warnings

# Suppress warning messages
# warnings.filterwarnings("ignore")

# Initialize doc
opt = docopt(__doc__)

# Define main function
def main(input_path, output_path):
    """
    Creates and saves all tables and figures from the EDA

    Parameters
    ----------
    input_path : string
        The input directory that contains the training data
    output_path : string
        The name of the directory that will contain the EDA plots and tables
    
    """
    # Check if the EDA directory exists; if it doesn't, create new folder.
    try:
        isDirExist = os.path.isdir(output_path)
        if not isDirExist:
            print("Directory does not exist. Creating a new folder...")
            os.makedirs(output_path)
            os.makedirs(f"{output_path}/figures/")
            os.makedirs(f"{output_path}/tables/")
    
    except Exception as ex:
        print("Exception occurred :" + ex)
        exit()

    # Verify that training data has been loaded
    isTrainExist = os.path.exists(f"{input_path}/airbnb_train_df.csv")
    if not isTrainExist:
        print('Training data has not been added.')
        exit()

    # read in the training data
    print('Training data has been partitioned.')
    train_df = pd.read_csv(f"{input_path}/airbnb_train_df.csv")

    # Get summary statistics from numeric columns
    summary_stats = train_df.drop(columns=['id', 'host_id']).describe()
    summary_stats.to_csv(f"{output_path}/tables/summary_stats.csv")

    # Generate density plots - currently does not save
    # aly.alt.data_transformers.enable('data_server')
    # density_plots = aly.dist(train_df.drop(columns=['id', 'name', 'host_name', 'neighbourhood_group', 'neighbourhood', 'room_type']))
    # save_chart(density_plots, f"{output_path}/figures/density_plots.png", 2)

    # Generate correlation matrix
    corr_matrix = train_df.drop(columns=['id']).corr('spearman')
    corr_matrix.to_csv(f"{output_path}/tables/correlation_matrix.csv")

    # Plot room types
    train_df['room_type'].value_counts().plot(kind='barh', title = 'Room Types')
    plt.xlabel('Count of Records')
    plt.savefig(f'{output_path}/figures/room_types.png')

    # Plot neighbourhoods
    train_df['neighbourhood'].value_counts()[:10].sort_values().plot(kind='barh', title = 'Top 10 Neighbourhoods');
    plt.xlabel('Count of Records')
    plt.savefig(f'{output_path}/figures/neighbourhoods.png')

    # Plot neighbourhood groups
    train_df['neighbourhood_group'].value_counts().sort_values().plot(kind='barh', title = 'Neighbour Groups');
    plt.xlabel('Count of Records')
    plt.savefig(f'{output_path}/figures/neighbourhood_groups.png')

    # Get most common words in the names of listings
    common_words = train_df['name'].str.split(' ').explode().value_counts()[:20].rename_axis('word').reset_index(name='count')
    common_words.to_csv(f"{output_path}/tables/common_words.csv")

# def save_chart(chart, filename, scale_factor=1):
#     """
#     Save an Altair chart using vl-convert
    
#     Parameters
#     ----------
#     chart : altair.Chart
#         Altair chart to save
#     filename : str
#         The path to save the chart to
#     scale_factor: int or float
#         The factor to scale the image resolution by.
#         E.g. A value of `2` means two times the default resolution.
#     """
#     if filename.split('.')[-1] == 'svg':
#         with open(filename, "w") as f:
#             f.write(vlc.vegalite_to_svg(chart.to_dict()))
#     elif filename.split('.')[-1] == 'png':
#         with open(filename, "wb") as f:
#             f.write(vlc.vegalite_to_png(chart.to_dict(), scale=scale_factor))
#     else:
#         raise ValueError("Only svg and png formats are supported")

# Call main
if __name__ == "__main__":
    main(opt["--input_path"], opt["--output_path"])