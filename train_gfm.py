from os.path import join
import numpy as np
import pandas as pd
import lightgbm as lgb
import pickle
from sklearn.metrics import mean_squared_error
from skopt import BayesSearchCV

import utils
import constants
import metrics
from add_features import add_time_lags, add_calendar_features, add_aggregations
from plots import plot_actuals_vs_predictions

def bayesian_hyperparam_search(
    X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
) -> lgb.LGBMRegressor:
    
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    param_space = {
        'num_leaves': (20, 50, 70),
        'learning_rate': (0.01, 0.1, 'log-uniform'),
        'feature_fraction': (0.7, 1.0, 'uniform'),
        'bagging_fraction': (0.7, 1.0, 'uniform'),
        'max_depth': (3, 10, 15),
        'min_child_samples': (5, 30)
    }

    # Set up bayesian optimisation
    opt = BayesSearchCV(
        estimator=lgb.LGBMRegressor(objective='regression', metric='mean_squared_error', boosting_type='gbdt'),
        search_spaces=param_space,
        n_iter=50,
        cv=3,
        n_jobs=-1,
        scoring='neg_mean_squared_error',
        verbose=1,
        random_state=42
    )

    opt.fit(X_train, y_train)

    best_estimator = opt.best_estimator_

    y_pred = best_estimator.predict(X_test, num_iteration=best_estimator.best_iteration)

    mse = mean_squared_error(y_test, y_pred)

    print(f"Best parameters found: {opt.best_params_}")
    print(f"Mean Squared Error on test set: {mse}")

    return opt.best_params_

def _train_test_split(raw_dataset: pd.DataFrame, train_fraction: float, country_code: int = None) -> tuple[pd.Series]:

    #train_size = int(len(y) * 0.8)

    #X_train, X_test = X[:train_size], X[train_size:]
    #y_train, y_test = y[:train_size], y[train_size:]

    # If more than one country, split the data so that each country is equally represented
    # in both training and testing sets
    train_data = pd.DataFrame()
    test_data = pd.DataFrame()

    if country_code:
        country_numerical_codes = [constants.COUNTRY_NUMERICAL_CODES_MAP[country_code]]
    else:
        # Get all countries
        country_numerical_codes = constants.COUNTRY_NUMERICAL_CODES_MAP.values()

    for code in country_numerical_codes:
        country_data = df[df['Country_Numerical_Code'] == code]
        train_size = int(len(country_data) * train_fraction)
        
        # Append to full datasets
        train_data = pd.concat([train_data, country_data.iloc[:train_size]])
        test_data = pd.concat([test_data, country_data.iloc[train_size:]])
    
    print(train_data.columns)
    
    X_train = train_data.drop(columns=['Wind_Energy_Potential'])
    y_train = train_data['Wind_Energy_Potential']

    X_test = test_data.drop(columns=['Wind_Energy_Potential'])
    y_test = test_data['Wind_Energy_Potential']
    
    X_train.astype(float)
    X_test.astype(float)
    y_train.astype(float)
    y_test.astype(float)

    return X_train, X_test, y_train, y_test

def _perform_algorithmic_partitioning():
    '''Instead of training a global forecasting model on the entire dataset,
    partitioning it into smaller almost equally sized parts has been shown
    to improve model performance. I could perform this partition on essentially
    any feature or combination of features (for example, grouping by countries
    whose time series have been shown to be strongly correlated), but a better
    solution is to perform a clustering approach such as k-means to algorithmically
    find a more effective way to partition the dataset.

    To effictively cluster time series we first have to extract richer time series
    characteristics (autocorrelation, mean, variance, peak-to-peak distance, entropy
    etc.) to allow the model to compare the distances between them. This can be done
    automatically with tsfel (time series feature extraction library).

    Charu C. Agarwal showed in 2001 that euclidian distance and other common distance
    metrics are often not effective in high-dimensional space, so instead of performing
    k-means directly on our tsfel feature dataset (which contains dozens of features),
    we should first reduce the dimensionality and perform k-means on the lower-dimensional
    space.

    A good way to do this is to use t-SNE, which projects the higher dimensional points into
    a lower dimensional space whilst trtying to maintain the distribution of distance between
    each point in the original space.

    Once we have a few clusters, we can separate the dataset and train a gfm for each of them,
    which should improve performance.
    '''
    pass

def train_gfm(
    X_train: pd.Series,
    X_test: pd.Series,
    y_train: pd.Series,
    y_test: pd.Series,
    save_as_filename: str
) -> None:
    
    # Find best hyperparams using bayesian optimisation
    #best_params = bayesian_hyperparam_search(
    #    X_train.values, y_train.values, X_test.values, y_test.values
    #)

    #print(best_params)
    
    # Retrain with the best params for more iterations
    best_params = {
        'boosting_type': 'gbdt',
        'bagging_fraction': 0.7629144166677786,
        'feature_fraction': 1.0,
        'learning_rate': 0.1,
        'max_depth': 15,
        'min_child_samples': 25,
        'num_leaves': 70
    }

    num_iters = 1000

    # Convert dataset for lgb
    train_data = lgb.Dataset(X_train.values, label=y_train.values)
    test_data = lgb.Dataset(X_test.values, label=y_test.values, reference=train_data)

    estimator = lgb.train(
        best_params,
        train_data,
        num_iters,
        valid_sets=[test_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=3),
        ])
    
    # Save model object
    utils.save_pickle(estimator, join('models', save_as_filename))

    y_pred: np.ndarray = estimator.predict(X_test, num_iteration=estimator.best_iteration)

    # Convert predictions to a pandas Series with the same index as y_test
    y_pred = pd.Series(y_pred, index=y_test.index)

    # Calculate residuals
    residuals = y_pred - y_test

    # Also calculate in-sample predictions for metric calculation
    y_train_pred = estimator.predict(X_train, num_iteration=estimator.best_iteration)

    # Display performance metrics and charts
    train_mse = metrics.mse(y_train_pred, y_train.values)
    test_mse = metrics.mse(y_pred, y_test.values)

    bias = metrics.bias(y_train_pred, y_train)
    variance = metrics.variance(y_pred)

    print(f'Train MSE: {train_mse:.6f}. Test MSE: {test_mse:.4f}')
    print(f'Bias: {bias:.6f}. Variance: {variance:.4f}')

def test_gfm(X_test: pd.Series, y_test: pd.Series, model_filename: str):

    estimator = utils.load_pickle(join('models', model_filename))

    y_pred: np.ndarray = estimator.predict(X_test, num_iteration=estimator.best_iteration)
    y_pred = pd.Series(y_pred, index=y_test.index)

    plot_actuals_vs_predictions(y_pred, y_test, last_n_days=7)

if __name__ == '__main__':

    df = pd.read_csv(join('data', 'all_countries_cleaned_30_years.csv'))
    df['Date'] = pd.to_datetime(df['Date'])     
    
    # Add features
    df = add_time_lags(df, target_key='Wind_Energy_Potential')
    df = add_calendar_features(df, datetime_feature='Date', num_fourier_terms=2)
    df = add_aggregations(df)

    # Remove Date feature before creating training and test set
    df = df.drop('Date', axis=1)
    
    _perform_algorithmic_partitioning()
    
    # Split up the dataset
    #X = df.drop(columns=['Wind_Energy_Potential'])
    #y = df['Wind_Energy_Potential']

    #X_train, X_test, y_train, y_test = _train_test_split(raw_dataset=df, train_fraction=0.95)

    #print(X_train.shape)
    #print(X_test.shape)
    #print(y_train.shape)
    #print(y_test.shape)
    
    #train_gfm(X_train, X_test, y_train, y_test, save_as_filename='lightgbm_global.pkl')
    
    _, X_test_fr, _, y_test_fr = _train_test_split(raw_dataset=df, train_fraction=0.95, country_code='PL')
    
    # Add datetime index back. TODO: Make this cleaner
    t = pd.date_range('1/1/1986', periods = len(X_test_fr.index), freq = 'h')
    X_test_fr.index = t
    y_test_fr.index = t
    
    test_gfm(X_test_fr, y_test_fr, 'lightgbm_global.pkl')