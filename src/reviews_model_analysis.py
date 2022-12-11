# author: Lauren Zung
# date: 2022-12-10

"""
Takes the preprocessed training and test data and tests multiple regressors.
Finds LGBMRegressor to be the best model.
 
Usage: src/reviews_model_analysis.py --input_path=data/processed/ --output_path=results/

Options:
--input_path=<input_path>           Path to the training data (in data/preprocessed)
--output_path=<output_path>         Path to the output directory for the tables and plots (results/)

Command to run the script:
python src/reviews_model_analysis.py --input_path=data/processed/ --output_path=results/
"""
import os
from docopt import docopt
import pandas as pd
import numpy as np
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer, MinMaxScaler, QuantileTransformer
from sklearn.feature_extraction import text 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import cross_validate
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform, randint, uniform
from sklearn.linear_model import LassoCV
from sklearn.tree import DecisionTreeRegressor
from lightgbm.sklearn import LGBMRegressor
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import eli5
import shap
from sklearn.metrics import mean_squared_error
import altair as alt
import vl_convert as vlc

opt = docopt(__doc__)

# Render figures given large data set
alt.data_transformers.disable_max_rows()

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
     # Verify that results directory exists; if not, creates a new folder
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

    # Verify that train and test data have been loaded
    isTrainExist = os.path.exists(f"{input_path}/airbnb_train_df.csv")
    if not isTrainExist:
        print('Training data has not been added.')
        exit()

    isTestExist = os.path.exists(f"{input_path}/airbnb_test_df.csv")
    if not isTestExist:
        print('Testing data has not been added')
        exit()
    
    # IMPORT DATA AND SPLIT-----------------------------------------------------------------------
    # read train and test data from csv files
    print("Reading data from CSV files...")
    train_df = pd.read_csv(f"{input_path}/airbnb_train_df.csv", na_filter=False)
    test_df = pd.read_csv(f"{input_path}/airbnb_test_df.csv", na_filter=False)
    
    # Separate X and y
    X_train = train_df.drop(columns='reviews_per_month')
    y_train = train_df['reviews_per_month']
    X_test = test_df.drop(columns='reviews_per_month')
    y_test = test_df['reviews_per_month']

    # PREPROCESSING-------------------------------------------------------------------------------
    
    # Assign feature types
    categorical_features = ['neighbourhood_group', 'neighbourhood', 'room_type']
    location_features = ['latitude', 'longitude']
    availability_feature = ['availability_365']
    numeric_features = ['price', 'minimum_nights', 'calculated_host_listings_count',
                        'n_words', 'vader_sentiment', 'rel_char_len']
    text_feature = 'name'
    drop_features = ['id', 'host_id', 'number_of_reviews', 'host_name', 'last_review']

    # Get unique neighbourhoods
    nyc_data = pd.read_csv('./data/raw/AB_NYC_2019.csv')
    neighbourhoods = nyc_data['neighbourhood'].unique().tolist()
    neighbourhoods = [x.lower() for x in neighbourhoods]
    neighbourhoods = [word for line in neighbourhoods for word in line.split()]

    # add extra words that aren't parsed:
    neighbourhoods = neighbourhoods + ['bedford', 'bull', 'hell', 'lefferts', 'op', 'prince']

    # Add to English stop words
    stop_words = text.ENGLISH_STOP_WORDS.union(neighbourhoods)

    # Create new ColumnTransformer
    preprocessor = make_column_transformer(
        (OneHotEncoder(sparse=True, handle_unknown='ignore', min_frequency=500), categorical_features),
        (KBinsDiscretizer(n_bins=20, encode='onehot'), location_features),
        (KBinsDiscretizer(n_bins=3, encode='onehot'), availability_feature),
        (QuantileTransformer(), numeric_features),
        (CountVectorizer(ngram_range=(1,3), max_features=100, stop_words=stop_words), text_feature),
        ('drop', drop_features)
    )

    # MODEL SELECTION-----------------------------------------------------------------------------
    
    # Initialize results dictionary
    cross_val_results = {}

    # Train a DummyRegressor
    print('DummyRegressor')
    dr = DummyRegressor()
    cross_val_results['DummyRegressor'] = pd.DataFrame(cross_validate(dr, X_train, y_train,
                                                       return_train_score=True)).agg(['mean', 'std']).round(3).T
    
    # Setup the Ridge pipeline
    print('Ridge')
    ridge_pipe = make_pipeline(preprocessor, Ridge())
    cross_val_results['Ridge'] = pd.DataFrame(cross_validate(ridge_pipe, X_train, y_train,
                                              return_train_score=True)).agg(['mean', 'std']).round(3).T
    
    # Setup the Lasso pipeline; set tolerance and iterations to limit warning messages
    print('Lasso')
    lasso_pipe = make_pipeline(preprocessor, LassoCV(tol=0.01, max_iter=10_000))
    cross_val_results['LassoCV'] = pd.DataFrame(cross_validate(lasso_pipe, X_train, y_train,
                                                return_train_score=True)).agg(['mean', 'std']).round(3).T

    # Setup decision tree pipeline; set max_depth and min_samples_leaf to prevent severe overfitting
    print('Decision Tree')
    dt_pipe = make_pipeline(preprocessor, DecisionTreeRegressor(max_depth=30, min_samples_leaf=100, random_state=123))
    cross_val_results['DecisionTreeRegressor'] = pd.DataFrame(cross_validate(dt_pipe, X_train, y_train,
                                                                            return_train_score=True)).agg(['mean', 'std']).round(3).T

    # Set up LGBM pipeline; set max_depth to prevent severe overfitting
    print('LGBMRegressor')
    lgbm_pipe = make_pipeline(preprocessor, LGBMRegressor(max_depth=30, random_state=123))
    cross_val_results['LightGBM'] = pd.DataFrame(cross_validate(lgbm_pipe, X_train, y_train,
                                                                return_train_score=True)).agg(['mean', 'std']).round(3).T

    # FEATURE SELECTION---------------------------------------------------------------------------
    print('LGBM L1 selection')
    # Setup L1 regularization pipeline to select features; set tolerance and maximum iterations to suppress warnings
    lgbm_l1 = make_pipeline(preprocessor, SelectFromModel(LassoCV(max_iter=10_000, tol=0.01)), LGBMRegressor(max_depth=30, random_state=123))

    cross_val_results['LGBM L1'] = pd.DataFrame(cross_validate(lgbm_l1, X_train, y_train, cv=10,
                                                               return_train_score=True)).agg(['mean', 'std']).round(3).T
    
    # HYPERPARAMETER TUNING-----------------------------------------------------------------------
    
    # Search for optimal alpha value
    print('Ridge HP Search')
    alpha_search = RandomizedSearchCV(ridge_pipe,
                                    {"ridge__alpha": loguniform(1e-3, 1e5)},
                                    verbose=1,
                                    n_jobs=-1,
                                    random_state=123,
                                    n_iter=20,
                                    return_train_score=True)
    alpha_search.fit(X_train, y_train);
    alpha_df = pd.DataFrame(alpha_search.cv_results_)[['mean_train_score', 'mean_test_score', 'param_ridge__alpha', 'rank_test_score']].set_index("rank_test_score").sort_index()
    alpha_df.to_csv(f"{output_path}/tables/ridge_alpha_search.csv")

    cross_val_results['Ridge Tuned'] = pd.DataFrame(cross_validate(alpha_search.best_estimator_, X_train, y_train,
                                                                   return_train_score=True)).agg(['mean', 'std']).round(3).T

    # Set up parameter distribution for LGBM
    param_dist = {"columntransformer__onehotencoder__max_categories": randint(low=5, high=100),
                "columntransformer__countvectorizer__max_features": randint(low=100, high=150),
                "lgbmregressor__max_depth": np.arange(5, 30, 5),
                "lgbmregressor__num_leaves": np.arange(5, 50, 5),
                "lgbmregressor__min_child_samples": np.arange(50, 300, 50),
                "lgbmregressor__reg_alpha": loguniform(1e-3, 1e6),
                "lgbmregressor__subsample": uniform()}

    # Hyperparameter tuning for LGBM
    print('LGBM HP Search')
    lgbm_search = RandomizedSearchCV(lgbm_pipe,
                                    param_dist,
                                    verbose=1,
                                    n_jobs=-1,
                                    random_state=123,
                                    n_iter=50,
                                    return_train_score=True)
    lgbm_search.fit(X_train, y_train);
    lgbm_search_df = pd.DataFrame(lgbm_search.cv_results_)[['mean_train_score', 'mean_test_score',
                                                            'param_columntransformer__onehotencoder__max_categories',
                                                            'param_columntransformer__countvectorizer__max_features',
                                                            'param_lgbmregressor__max_depth',
                                                            'param_lgbmregressor__num_leaves',
                                                            'param_lgbmregressor__min_child_samples',
                                                            'param_lgbmregressor__reg_alpha',
                                                            'param_lgbmregressor__subsample',
                                                            'rank_test_score']].set_index("rank_test_score").sort_index()[:10]
    lgbm_search_df = lgbm_search_df.rename(columns={'param_columntransformer__onehotencoder__max_categories': 'max_categories',
                                                    'param_columntransformer__countvectorizer__max_features': 'max_features',
                                                    'param_lgbmregressor__max_depth': 'max_depth',
                                                    'param_lgbmregressor__num_leaves': 'num_leaves',
                                                    'param_lgbmregressor__min_child_samples': 'min_child_samples',
                                                    'param_lgbmregressor__reg_alpha': 'reg_alpha',
                                                    'param_lgbmregressor__subsample': 'subsample'})
    lgbm_search_df.to_csv(f"{output_path}/tables/lgbm_search.csv")

    # CV results of the best LGBM
    cross_val_results['LGBM Tuned'] = pd.DataFrame(cross_validate(lgbm_search.best_estimator_, X_train, y_train,
                                                                return_train_score=True)).agg(['mean', 'std']).round(3).T

    # Set up parameter distribution for decision tree regressor
    param_dist = {"columntransformer__onehotencoder__max_categories": randint(low=5, high=100),
                "columntransformer__countvectorizer__max_features": randint(low=20, high=200),
                "decisiontreeregressor__max_depth": randint(low=5, high=40),
                "decisiontreeregressor__criterion": ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                "decisiontreeregressor__min_samples_leaf": loguniform(1e-5, 1e-1),
                "decisiontreeregressor__max_features": [0.25, 0.5, 0.75, 'sqrt', 'log2'],
                "decisiontreeregressor__ccp_alpha": loguniform(1e-3, 1e3)}

    # Hyperparameter tuning
    print('Decision Tree HP Search')
    dt_search = RandomizedSearchCV(dt_pipe,
                                    param_dist,
                                    verbose=1,
                                    n_jobs=-1,
                                    random_state=123,
                                    n_iter=50,
                                    return_train_score=True)
    dt_search.fit(X_train, y_train);
    dt_search_df = pd.DataFrame(dt_search.cv_results_)[['mean_train_score', 'mean_test_score',
                                                        'param_columntransformer__onehotencoder__max_categories',
                                                        'param_columntransformer__countvectorizer__max_features',
                                                        'param_decisiontreeregressor__max_depth',
                                                        'param_decisiontreeregressor__criterion',
                                                        'param_decisiontreeregressor__min_samples_leaf',
                                                        'param_decisiontreeregressor__max_features',
                                                        'param_decisiontreeregressor__ccp_alpha',
                                                        'rank_test_score']].set_index("rank_test_score").sort_index()[:10]
    dt_search_df = dt_search_df.rename(columns={'param_columntransformer__onehotencoder__max_categories': 'max_categories',
                                                'param_columntransformer__countvectorizer__max_features': 'max_features',
                                                'param_decisiontreeregressor__max_depth': 'max_depth',
                                                'param_decisiontreeregressor__criterion': 'criterion',
                                                'param_decisiontreeregressor__min_samples_leaf': 'min_samples_leaf',
                                                'param_decisiontreeregressor__max_features': 'max_features',
                                                'param_decisiontreeregressor__ccp_alpha': 'ccp_alpha'})
    dt_search_df.to_csv(f"{output_path}/tables/dt_search.csv")

    # CV results of the best decision tree
    cross_val_results['DT Tuned'] = pd.DataFrame(cross_validate(dt_search.best_estimator_, X_train, y_train,
                                                                return_train_score=True)).agg(['mean', 'std']).round(3).T

    # Hyperparameter tuning for lasso
    print('Lasso HP Search')
    las_tol_search = RandomizedSearchCV(lasso_pipe,
                                        {"lassocv__tol": loguniform(1e-3, 1e1)},
                                        verbose=1,
                                        n_jobs=-1,
                                        random_state=123,
                                        n_iter=20,
                                        return_train_score=True)
    las_tol_search.fit(X_train, y_train);
    las_tol_df = pd.DataFrame(las_tol_search.cv_results_)[['mean_train_score', 'mean_test_score', 'param_lassocv__tol', 'rank_test_score']].set_index("rank_test_score").sort_index()
    las_tol_df.to_csv(f"{output_path}/tables/lasso_tol_search.csv")

    # CV results of the best Lasso
    cross_val_results['Lasso Tuned'] = pd.DataFrame(cross_validate(las_tol_search.best_estimator_, X_train, y_train,
                                                                return_train_score=True)).agg(['mean', 'std']).round(3).T

    # FEATURE IMPORTANCES---------------------------------------------------------------------

    # Sort examples into above and below the median number of reviews
    y_train = y_train.reset_index(drop=True)
    y_train_lower = y_train[y_train < np.median(y_train)]
    y_train_higher = y_train[y_train >= np.median(y_train)]

    X_train_enc = lgbm_search.best_estimator_.named_steps['columntransformer'].fit_transform(X_train)
    column_names = lgbm_search.best_estimator_.named_steps['columntransformer'].get_feature_names_out().tolist()
    X_train_enc_df = pd.DataFrame(X_train_enc.toarray(), columns=column_names, index=X_train.index)

    X_train_lower = X_train_enc_df.loc[X_train_enc_df.index.isin(y_train_lower.index)]
    X_train_higher = X_train_enc_df.loc[X_train_enc_df.index.isin(y_train_higher.index)]

    # Get eli5 figure
    weights_df = eli5.explain_weights_df(lgbm_search.best_estimator_.named_steps["lgbmregressor"], feature_names=column_names)
    weights_df.to_csv(f"{output_path}/tables/lgbm_eli5_weights.csv")

    # Create a shap explainer object 
    lgbm_explainer = shap.TreeExplainer(lgbm_search.best_estimator_.named_steps["lgbmregressor"])
    train_lgbm_shap_values = lgbm_explainer.shap_values(X_train_enc_df)
    shap.summary_plot(train_lgbm_shap_values, X_train_enc_df)
    plt.savefig(f'{output_path}/figures/shap_summary_plot.png')

    # Below median example
    shap.force_plot(
        lgbm_explainer.expected_value, # expected value 
        train_lgbm_shap_values[X_train_lower.index[1291]], # SHAP values associated with an example with lower reviews per month
        X_train_lower.iloc[1291, :], # Feature vector of the example 
        matplotlib=True,
        text_rotation=45
    )
    plt.savefig(f'{output_path}/figures/train_shap_below_median_force_plot.png')

    # Above median example
    shap.force_plot(
        lgbm_explainer.expected_value, # expected value 
        train_lgbm_shap_values[X_train_higher.index[2192]], # SHAP values associated with an example with higher reviews per month
        X_train_higher.iloc[2192, :], # Feature vector of the example 
        matplotlib=True,
        text_rotation=45
    )
    plt.savefig(f'{output_path}/figures/train_shap_above_median_force_plot.png')

    # Highest number of reviews in the training data
    shap.force_plot(
        lgbm_explainer.expected_value, # expected value 
        train_lgbm_shap_values[np.argmax(y_train)], # SHAP values associated with the example with the highest number of reviews per month
        X_train_enc_df.iloc[np.argmax(y_train), :], # Feature vector of the example 
        matplotlib=True,
        text_rotation=45
    )
    plt.savefig(f'{output_path}/figures/train_shap_most_reviews_force_plot.png')

    # TEST SET RESULTS-----------------------------------------------------------------------
    # Compute test R^2 and RMSE scores
    results = {}
    results['Train R^2'] = lgbm_search.best_estimator_.score(X_train, y_train)
    results['Train RMSE'] = mean_squared_error(y_train, lgbm_search.best_estimator_.predict(X_train))**0.5
    results['Test R^2'] = lgbm_search.best_estimator_.score(X_test, y_test)
    results['Test RMSE'] = mean_squared_error(y_test, lgbm_search.best_estimator_.predict(X_test))**0.5
    train_test_results = pd.DataFrame(results.values(), columns=['Score'], index=results.keys())
    train_test_results.to_csv(f"{output_path}/tables/train_test_results.csv")

    # Sort examples into above and below the median number of reviews
    y_test = y_test.reset_index(drop=True)
    y_test_lower = y_test[y_test < np.median(y_test)]
    y_test_higher = y_test[y_test >= np.median(y_test)]

    X_test_enc = lgbm_search.best_estimator_.named_steps['columntransformer'].transform(X_test)
    X_test_enc_df = pd.DataFrame(X_test_enc.toarray(), columns=column_names, index=X_test.index)

    X_test_lower = X_test_enc_df.loc[X_test_enc_df.index.isin(y_test_lower.index)]
    X_test_higher = X_test_enc_df.loc[X_test_enc_df.index.isin(y_test_higher.index)]

    # Create a shap explainer object 
    test_lgbm_shap_values = lgbm_explainer.shap_values(X_test_enc_df)

    # Below median example
    shap.force_plot(
        lgbm_explainer.expected_value, # expected value 
        test_lgbm_shap_values[X_test_lower.index[3]], # SHAP values associated with an example with lower reviews per month
        X_test_lower.iloc[0, :], # Feature vector of the example 
        matplotlib=True,
        text_rotation=45
    )
    plt.savefig(f'{output_path}/figures/test_shap_below_median_force_plot.png')

    # Above median example
    shap.force_plot(
        lgbm_explainer.expected_value, # expected value 
        test_lgbm_shap_values[X_test_higher.index[0]], # SHAP values associated with an example with higher reviews per month
        X_test_higher.iloc[0, :], # Feature vector of the example 
        matplotlib=True,
        text_rotation=45
    )
    plt.savefig(f'{output_path}/figures/test_shap_above_median_force_plot.png')

    # RESULTS SUMMARY-----------------------------------------------------------------------
    # Compile all model results
    cross_val_results_table = pd.concat(cross_val_results, axis=1).T
    cross_val_results_table.to_csv(f"{output_path}/tables/cross_val_results.csv")

    # Plot predicted vs actual reviews per month
    plt.plot(y_test, y_test, color='r')
    plt.scatter(lgbm_search.best_estimator_.predict(X_test), y_test, alpha=0.2)
    plt.xlabel('Predicted Reviews per Month')
    plt.ylabel('Actual Reviews per Month')
    plt.savefig(f'{output_path}/figures/pred_vs_actual.png')

    # Plot predicted vs actual reviews per month, zoomed into majority of samples
    plt.plot(y_test, y_test, color='r')
    plt.scatter(lgbm_search.best_estimator_.predict(X_test), y_test, alpha=0.2)
    plt.xlabel('Predicted Reviews per Month')
    plt.ylabel('Actual Reviews per Month')
    plt.ylim(0,10)
    plt.xlim(0,10)
    plt.savefig(f'{output_path}/figures/pred_vs_actual_zoomed.png')

    # Create a heatmap to avoid overplotting
    pred_vs_act = pd.DataFrame({'pred': lgbm_search.best_estimator_.predict(X_test), 'act': y_test})

    pred_vs_act_chart = (
        alt.Chart(pred_vs_act).mark_rect(clip=True).encode(
            alt.X('pred', title='Predicted Reviews per Month', bin=alt.Bin(maxbins=30), scale=alt.Scale(domain=(0, 7))),
            alt.Y('act', title='Actual Reviews per Month', bin=alt.Bin(maxbins=30), scale=alt.Scale(domain=(0, 20))),
            alt.Color('count()'))
    )
    save_chart(pred_vs_act_chart, f'{output_path}/figures/pred_vs_actual_heatmap.png', 2)


def save_chart(chart, filename, scale_factor=1):
    """
    Save an Altair chart using vl-convert
    
    Parameters
    ----------
    chart : altair.Chart
        Altair chart to save
    filename : str
        The path to save the chart to
    scale_factor: int or float
        The factor to scale the image resolution by.
        E.g. A value of `2` means two times the default resolution.
    """
    if filename.split('.')[-1] == 'svg':
        with open(filename, "w") as f:
            f.write(vlc.vegalite_to_svg(chart.to_dict()))
    elif filename.split('.')[-1] == 'png':
        with open(filename, "wb") as f:
            f.write(vlc.vegalite_to_png(chart.to_dict(), scale=scale_factor))
    else:
        raise ValueError("Only svg and png formats are supported")

# Call main
if __name__ == "__main__":
    main(opt["--input_path"], opt["--output_path"])