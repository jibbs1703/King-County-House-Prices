from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from training.helpers.aws_services import S3Buckets
import pandas as pd
import yaml

with open('training/config.yaml', 'r') as file:
    train_yaml = yaml.safe_load(file)


class ModelInputs:
    def __init__(self, df):
        self.df = df

    def target_feature_split(self):
        X = self.df.drop(train_yaml['TARGET'], axis = 'columns')
        y = self.df[train_yaml['TARGET']]
        return X, y

    def feature_split(self, X, y, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=420)
        return X_train, X_test, y_train, y_test

    def run_pipeline(self):
        X, y = self.target_feature_split()
        X_train, X_test, y_train, y_test = self.feature_split(X, y, test_size=train_yaml['TEST_SIZE'])
        return X_train, X_test, y_train, y_test


class ModelTrain:
    @staticmethod
    def model_training(X_train, y_train):
        # Instantiate Model and Fit to Training Data
        algorithm = HistGradientBoostingRegressor(random_state=420)
        algorithm.fit(X_train, y_train)

        # Save the Fitted Model To S3 Bucket
        s3 = S3Buckets.credentials('us-east-2')
        s3.save_model_to_s3(algorithm, train_yaml['MODEL_BUCKET'], train_yaml['MODEL_NAME'])

        # Return the Fitted Model
        return algorithm


class ModelPredict:
    @staticmethod
    def model_prediction(model, X_test, y_test):
        # Get Prediction on the Test Data
        y_test_pred = model.predict(X_test)
        # Transform Predicted Values to Same Format as Actual Values
        y_test_pred = pd.Series(y_test_pred)
        y_test_pred.index = y_test.index
        return y_test_pred
