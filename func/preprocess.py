import numpy as np
import pandas as pd
import category_encoders as ce
from IPython.display import display
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

def feature_wrangler(df):
    '''This function examines the columns of the House Prices DataFrame, performs some feature enginnering on the columns and and then returns the cleaned dataframe, returning the top 10 columns of the cleaned dataframe.
    :param : Uncleaned DataFrame
    :returns: Cleaned DataFrame'''

    # Set the Index to the id Column and Drop the id Column
    df.index = df['id']
    df = df.drop(columns=['id', 'date', 'lat', 'long'], axis=1)

    # Replace Missing Values in Dataset According to Datatypes (Mode for Categories. Mean for Continuous)
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].nunique() > 10:
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])

    # Set Datatypes For Categorical Features
    df['condition'] = df['condition'].astype(object)
    df['grade'] = df['grade'].astype(object)
    df['zipcode'] = df['zipcode'].astype(object)
    df['waterfront'] = df['waterfront'].astype(object)

    # Return Cleaned Dataset
    return df


def outlier_remover(df):
    '''
    This function filters out all column values above and below three standard deviations from the column mean.

    :param : DataFrame
    :returns: DataFrame with all columns within three standard deviations above and below their mean values.
    '''
    for col in df.columns:
        if df[col].dtype in ['int64', 'float64']:
            if col not in ['lat', 'long']:
                mean = df[col].mean()
                std = df[col].std()
                df = df[(df[col] > df[col].mean() - 3 * df[col].std()) & (df[col] < df[col].mean() + 3 * df[col].std())]

    # Return Dataset With Outliers Removed
    return df


def cat_encoder(df):
    '''
    This function encodes all categorical variables in the input dataframe using Target Encoding
    :param DataFrame
    :return: DataFrame with Target Encoded Categorical Features
    '''
    # Instantiate Target Encoder
    te = ce.TargetEncoder()

    # Extract the Categorical Columns
    cat_cols = list(df.select_dtypes(include=['object']).columns)

    # Target-Encode the Categorical Features
    for col in cat_cols:
        df[col] = te.fit_transform(df[col], df.price)

    # Return Dataset With Categorical Features Transformed
    return df


def vif_screener(df):
    '''
    This function drops all features with a variance inflation factor (VIF) above 5.
    It also displays the VIFs before and after the highly correlated features get dropped.
    :param: DataFrame
    :returns: DataFrame with highly correlated features dropped
    '''
    # Branch Out Dataframe into 2 Dataframes to Examine the VIF Before and After Highly Correlated Columns are Dropped
    d_before = df
    d_after = df

    # VIF Before Dropping Columns
    before_dropping = sm.tools.add_constant(d_before)
    series_before = pd.Series(
        [variance_inflation_factor(before_dropping.values, i) for i in range(before_dropping.shape[1])],
        index=before_dropping.columns)

    # Display the series
    print('VIF BEFORE DROPPING HIGHLY CORRELATED COLUMNS COLUMNS')
    print('-' * 55)
    display(series_before)

    # Define Columns to Drop based on VIF Threshold
    high_vif = series_before[series_before > 5]
    high_vif = high_vif.index.to_list()
    high_vif.remove('const')
    high_vif

    # VIF After Dropping Columns
    d_after = d_after.drop(columns=high_vif)
    after_dropping = sm.tools.add_constant(d_after)
    series_after = pd.Series(
        [variance_inflation_factor(after_dropping.values, i) for i in range(after_dropping.shape[1])],
        index=after_dropping.columns)

    # Display the series
    print('VIF AFTER DROPPING HIGHLY CORRELATED COLUMNS COLUMNS')
    print('-' * 55)
    display(series_after)

    # Collect Columns into New List and Create New Dataframe
    final_columns = series_after.index.to_list()
    final_columns.remove('const')
    final_columns

    df = df[final_columns]

    return df