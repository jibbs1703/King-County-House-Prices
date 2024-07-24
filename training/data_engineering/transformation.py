# Import Libraries to Run Module
import pandas as pd
import category_encoders as ce
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import yaml
import warnings
warnings.filterwarnings("ignore")

# Access config.yaml File to Access Saved Parameters
with open('training/config.yaml', 'r') as file:
    train_yaml = yaml.safe_load(file)


# Create Transformation Class to Carry Out Model Preprocessing and Feature Engineering
class Transformation:
    def __init__(self, df):
        self.df = df

    def feature_wrangler(self):
        """This function examines the columns of the House Prices DataFrame, performs
        some feature engineering on the columns and then returns the cleaned dataframe,
        returning the top 10 columns of the cleaned dataframe.
        :param : Uncleaned DataFrame
        :returns: Cleaned DataFrame
        """
        # Set the Index to the id Column and Drop the id Column
        self.df.index = self.df['id']
        self.df = self.df.drop(columns=train_yaml['INITIAL_DROP'], axis=1)

        # Replace Missing Values in Dataset According to Datatypes (Mode for Categories. Mean for Continuous)
        for col in self.df.columns:
            if self.df[col].isnull().sum() > 0:
                if self.df[col].nunique() > 10:
                    self.df[col] = self.df[col].fillna(self.df[col].mean())
                else:
                    self.df[col] = self.df[col].fillna(self.df[col].mode()[0])

        # Set Datatypes For Categorical Features
        self.df[train_yaml['CAT_COLUMNS']] = self.df[train_yaml['CAT_COLUMNS']].astype(str)

        # Set Datatypes For Numeric Features
        self.df[train_yaml['NUM_COLUMNS']] = self.df[train_yaml['NUM_COLUMNS']].apply(pd.to_numeric,errors='coerce', downcast='float')

    def outlier_remover(self):
        """
        This function filters out all column values above and below three standard deviations from the column mean.

        :param : DataFrame
        :returns: DataFrame with all Numeric columns within three standard deviations above and below their mean values.
        """
        for col in train_yaml['NUM_COLUMNS'] + train_yaml['TARGET']:
            mean = self.df[col].mean()
            std = self.df[col].std()
            self.df = self.df[(self.df[col] > self.df[col].mean() - 3 * self.df[col].std()) & (self.df[col] < self.df[col].mean() + 3 * self.df[col].std())]

    def cat_encoder(self):
        """
        This function encodes all categorical variables in the input dataframe using Target Encoding
        :return: DataFrame with Target Encoded Categorical Features
        """
        # Instantiate Target Encoder
        te = ce.TargetEncoder()

        # Extract the Categorical Columns
        cat_cols = train_yaml['CAT_COLUMNS']
        # Target-Encode the Categorical Features
        for col in cat_cols:
            self.df[col] = te.fit_transform(self.df[col], self.df[train_yaml['TARGET']])

    def vif_screener(self, vif_threshold=5):
        """
        This function drops all features with a variance inflation factor (VIF) above a specified threshold.
        The default VIF cut-off is 5
        :param: DataFrame
        :returns: DataFrame with highly correlated features dropped
        """
        # Branch Out Dataframe into 2 Dataframes to Examine the VIF Before and After Highly Correlated Columns are Dropped
        d_before = self.df
        d_after = self.df

        # VIF Before Dropping Columns
        before_dropping = sm.tools.add_constant(d_before)
        series_before = pd.Series(
            [variance_inflation_factor(before_dropping.values, i) for i in range(before_dropping.shape[1])],
            index=before_dropping.columns)

        # Define Columns to Drop based on VIF Threshold
        high_vif = series_before[series_before > vif_threshold]
        high_vif = high_vif.index.to_list()
        high_vif.remove('const')

        # VIF After Dropping Columns
        d_after = d_after.drop(columns=high_vif)
        after_dropping = sm.tools.add_constant(d_after)
        series_after = pd.Series(
            [variance_inflation_factor(after_dropping.values, i) for i in range(after_dropping.shape[1])],
            index=after_dropping.columns)

        # Collect Columns into New List and Create New Dataframe
        final_columns = series_after.index.to_list()
        final_columns.remove('const')

        # Filter out the High VIF Columns in the Dataframe
        self.df = self.df[final_columns + train_yaml['TARGET']]

    def run_pipeline(self):
        self.feature_wrangler()
        self.outlier_remover()
        self.cat_encoder()
        self.vif_screener(vif_threshold=train_yaml['VIF_THRESHOLD'])

        return self.df