import re
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_validate, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

df = pd.read_csv("hitters.csv")

# Exploratory Data Analysis

def classify_columns_types(dataframe, cat_th=10, car_th=20):
    """
    Classify columns in a DataFrame into categorical, numerical, and categorical but cardinal variables.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The DataFrame containing the columns to be classified.
    cat_th : int, optional
        The threshold value for the number of unique values in numerical-looking categorical variables.
    car_th : int, optional
        The threshold value for the number of unique values in categorical but cardinal variables.

    Returns
    -------
    tuple
        A tuple containing three lists:
        - cat_cols: List of categorical variables.
        - num_cols: List of numerical variables.
        - cat_but_car: List of categorical but cardinal variables.
    """
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    return cat_cols, num_cols, cat_but_car

def categorical_column_summary(dataframe, categorical_cols, target_col, plot=False):
    """
    Analyze categorical columns in a DataFrame.

    Parameters:
    - dataframe (pd.DataFrame): The input DataFrame.
    - categorical_cols (list of str): List of column names containing categorical data.
    - target_col (str): The name of the target column for which the mean is calculated.
    - plot (bool, optional): Whether to plot the data. Default is False.

    Returns:
    - None

    Prints:
    - Mean of the target column for each unique combination of categorical columns.
    - Count and ratio of each unique value in the categorical columns.
    """
    print(dataframe.groupby(categorical_cols)[target_col].mean())
    for col in categorical_cols:
        counts = dataframe[col].value_counts()
        ratios = 100 * counts / len(dataframe)
        print(pd.DataFrame({"Count": counts, "Ratio": ratios}))

        if plot:
            plt.figure()
            sns.countplot(x=col, data=dataframe)
            plt.show()

def numeric_column_summary(dataframe, numerical_cols, target_col, plot=False):
    """
    Analyze numerical columns in a DataFrame.

    Parameters:
    - dataframe (pd.DataFrame): The input DataFrame.
    - numerical_cols (list of str): List of column names containing numerical data.
    - target_col (str): The name of the target column used for grouping and calculating mean.
    - plot (bool, optional): Whether to plot histograms for each numerical column. Default is False.

    Returns:
    - None

    Prints:
    - Summary statistics (mean, quartiles) for each numerical column.
    - Mean of each numerical column grouped by quantiles of the target column.
    - Histograms (optional) for each numerical column.
    """
    quantiles = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    print(dataframe[numerical_cols].describe(quantiles).T)

    if dataframe[target_col].dtype == 'object':
        print(dataframe.groupby(target_col).agg({col: "mean" for col in numerical_cols}))
    else:
        for col in numerical_cols:
            print(dataframe.groupby(pd.qcut(dataframe[target_col], quantiles)).agg({col: "mean"}))

    if plot:
        for col in numerical_cols:
            plt.figure()
            dataframe[col].hist(bins=20)
            plt.xlabel(col)
            plt.title(col)
            plt.show()

def correlation_matrix():
    plt.figure(figsize=(60,60))
    corr=df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(df.corr(), mask=mask, cmap='coolwarm', vmax=.3, center=0,
                square=True, linewidths=.5,annot=True)
    plt.show(block=True)

# Data Preprocessing & Feature Engineering

def calculate_outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    """
    Calculate the lower and upper outlier thresholds based on the interquartile range (IQR).

    Parameters:
    - dataframe (pd.DataFrame): The input DataFrame.
    - col_name (str): The name of the column for which outliers are being calculated.
    - q1 (float, optional): The first quartile value. Default is 0.25.
    - q3 (float, optional): The third quartile value. Default is 0.75.

    Returns:
    - tuple: A tuple containing the lower and upper outlier thresholds.
    """

    q1_value, q3_value = dataframe[col_name].quantile([q1, q3])
    iqr = q3_value - q1_value
    return float(q1_value - 1.5 * iqr), float(q3_value + 1.5 * iqr)

def replace_outliers_with_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    """
    Replace outliers in a column of a DataFrame with the lower and upper thresholds.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The DataFrame containing the column with outliers.
    variable : str
        The name of the column with outliers.
    q1 : float, optional
        The first quartile value. Default is 0.25.
    q3 : float, optional
        The third quartile value. Default is 0.75.

    Returns
    -------
    None
    """
    low_limit, up_limit = calculate_outlier_thresholds(dataframe, col_name, q1, q3)
    dataframe.loc[dataframe[col_name] < float(low_limit), col_name] = int(low_limit)
    dataframe.loc[dataframe[col_name] > float(up_limit), col_name] = int(up_limit)

def plot_quartiles_outlier(dataframe, col_name):
    """
        Plot boxplots for each column in a list of column names.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            The DataFrame containing the columns to plot.
        col_name : list of str
            The list of column names to plot.

        Returns
        -------
        None
        """
    for col in col_name:
        plt.figure()
        sns.boxplot(data=dataframe[col])
        plt.show()

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    """
        Perform one-hot encoding on categorical columns of a DataFrame.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            The DataFrame containing the categorical columns to be encoded.
        categorical_cols : list of str
            The list of column names containing categorical data.
        drop_first : bool, optional
            Whether to drop the first encoded column for each categorical variable.
            Default is False.

        Returns
        -------
        pandas.DataFrame
            The DataFrame with one-hot encoded columns.
        """
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first, dtype=int)
    return dataframe

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe
def standardization(dataframe, numerical_cols, scaler):
    X_scaled = scaler.fit_transform(dataframe[numerical_cols])
    dataframe[numerical_cols] = pd.DataFrame(X_scaled, columns=dataframe[numerical_cols].columns)
    return dataframe[numerical_cols]


# Function to evaluate a model and print the results
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    # Train Error
    y_pred_train = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    train_r2 = model.score(X_train, y_train)

    # Test Error
    y_pred_test = model.predict(X_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_r2 = model.score(X_test, y_test)

    print(f"{model_name} Train RMSE: {train_rmse:.2f}")
    print(f"{model_name} Train R^2: {train_r2:.3f}")
    print(f"{model_name} Test RMSE: {test_rmse:.2f}")
    print(f"{model_name} Test R^2: {test_r2:.3f}")


# Function to tune hyperparameters of a model using GridSearchCV
def tune_model(model, param_grid, X_train, y_train, X_test, y_test, model_name):
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    print(f"Best {model_name} Parameters: {best_params}")

    evaluate_model(best_model, X_train, y_train, X_test, y_test, f"Tuned {model_name}")



# Feature Importance
def plot_importance(model, features, save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:len(features)])
    plt.title('Features')
    plt.tight_layout()
    plt.show(block=True)
    if save:
        plt.savefig(f'importances.png')



