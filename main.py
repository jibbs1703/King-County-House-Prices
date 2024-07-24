from training.helpers.data_reader import csv_loader
from training.helpers.aws_services import S3Buckets
from training.data_engineering.transformation import Transformation
from training.model.model import ModelInputs, ModelTrain, ModelPredict
from training.model.metrics import ModelMetrics
import yaml

with open('training/config.yaml', 'r') as file:
    train_yaml = yaml.safe_load(file)

s3 = S3Buckets.credentials('us-east-2')

# Upload Raw Dataset to Raw-Data Bucket if Not Already Uploaded
# uncleaned_kc_data = csv_loader('https://raw.githubusercontent.com/Amberlynnyandow/dsc-1-final-project-online-ds-ft-021119/master/kc_house_data.csv')
# s3.upload_dataframe_to_s3(df = uncleaned_kc_data, bucket_name= 'jibbs-raw-datasets', object_name='uncleaned_kc_housing_data.csv')

# Import Uncleaned Dataset From Raw-Data Bucket and Load as Dataframe
uncleaned_file = s3.read_file('jibbs-raw-datasets', 'uncleaned_kc_housing_data.csv')
data = csv_loader(uncleaned_file)

# Transformation and Feature Engineering
transform = Transformation(data)
data = transform.run_pipeline()

# Upload Cleaned Dataframe to S3 Bucket
s3.upload_dataframe_to_s3(data, 'jibbs-cleaned-datasets', 'cleaned_kc_housing_data.csv')

# Load Cleaned File From S3 Bucket
cleaned_file = s3.read_file('jibbs-cleaned-datasets', 'cleaned_kc_housing_data.csv')
data = csv_loader(cleaned_file)

# Create Training and Test Data from Features and Target
model_input = ModelInputs(data)
train_features, test_features, train_target, test_target = model_input.run_pipeline()

TRAIN = input("Do You want to Train the Model "
              "(T Trains Only and F Trains and Predicts on Test Data and Provides Metrics)?: ")

if TRAIN == 'T':
    model_train = ModelTrain()
    model = model_train.model_training(train_features, train_target)

else:
    model_train = ModelTrain()
    model = model_train.model_training(train_features, train_target)
    # model_pred = ModelPredict()
    test_prediction = ModelPredict.model_prediction(model, test_features, test_target)
    # test_prediction = model_pred.model_prediction(model, test_features, test_target)
    print(test_prediction.head())
    metrics = ModelMetrics(test_target, test_prediction)
    print(metrics.model_r2())
