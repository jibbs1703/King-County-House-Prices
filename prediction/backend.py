from typing import Literal
from fastapi import FastAPI, Depends
from pydantic import BaseModel, Field
import yaml
import pickle
import pandas as pd

# Access Config.yaml File For Training Package to Access Saved Parameters For Model Training
with open('config.yaml', 'r') as file:
    yaml_file = yaml.safe_load(file)

# Instantiate FastAPI
app = FastAPI(title="House Price Prediction Application", version='0.0.1')


# Description of Model Features
class Features(BaseModel):
    bedrooms: int = Field(title='Number of Bedrooms')
    bathrooms: float = Field(title='Number of Bathrooms')
    sqft_lot: int = Field(title='Lot Size (in square feet)')
    floors: int = Field(title='Number of Floors')
    view: int = Field(title='Number of Buyer Viewings')
    yr_built: int = Field(title='Year House was Built')
    yr_renovated: int = Field(title='Year the House was Renovated (Use Year House was Built if not renovated)')
    sqft_living15: int = Field(title='Home Size of Nearest 15 neighbors (in square feet)', default=1840)
    sqft_lot15: int = Field(title='Lot Size of Nearest 15 neighbors (in square feet)', default=7620)
    condition: Literal['1', '2', '3', '4', '5'] = Field(title='Condition of House')
    grade: Literal['3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13'] = Field(title='Grade of House')
    zipcode: Literal['98001', '98002', '98003', '98004', '98005', '98006', '98007', '98008', '98010', '98011', '98014', '98019', '98022', '98023', '98024', '98027', '98028', '98029', '98030', '98031', '98032', '98033', '98034', '98038', '98039', '98040', '98042', '98045', '98052', '98053', '98055', '98056', '98058', '98059', '98065', '98070', '98072', '98074', '98075', '98077', '98092', '98102', '98103', '98105', '98106', '98107', '98108', '98109', '98112', '98115', '98116', '98117', '98118', '98119', '98122', '98125', '98126', '98133', '98136', '98144', '98146', '98148', '98155', '98166', '98168', '98177', '98178', '98188', '98198', '98199'] = Field(title='Zipcode')


# Load Trained Model Only When App Starts-Up
@app.on_event("startup")
def model_encoder_scaler():
    # Load the Model, Encoder and Scaler
    loaded_scaler = pickle.load(open(f"objects/{yaml_file['NUM_SCALER']}", 'rb'))
    loaded_encoder = pickle.load(open(f"objects/{yaml_file['CAT_ENCODER']}", 'rb'))
    loaded_model = pickle.load(open(f"objects/{yaml_file['MODEL_NAME']}", 'rb'))

    return loaded_model, loaded_encoder, loaded_scaler


# Assign Loaded Model, Encoder and Scaler to Objects
model, encoder, scaler = model_encoder_scaler()


# Create Landing Page for Application
@app.get("/")
def home():
    return "House Price Prediction Application"


# Create Prediction Path
@app.post("/predict")
def predict(features: Features = Depends()):
    # Collect Input into a Dictionary and Convert to DataFrame
    data = features.dict()
    data = pd.DataFrame(data, index=[0])

    # Set Datatypes For Categorical and Numeric Features
    data[yaml_file['CAT_COLUMNS']] = data[yaml_file['CAT_COLUMNS']].astype(str)
    data[yaml_file['NUM_COLUMNS_SCALED']] = data[yaml_file['NUM_COLUMNS_SCALED']].apply(pd.to_numeric, errors='coerce', downcast='float')

    # Encode and Scale Features
    data[yaml_file['NUM_COLUMNS_SCALED']] = scaler.transform(data[yaml_file['NUM_COLUMNS_SCALED']])
    data[yaml_file['CAT_COLUMNS']] = encoder.transform(data[yaml_file['CAT_COLUMNS']])

    # Get Prediction From transformed Features
    prediction = model.predict(data[yaml_file['FEATURE_ORDER']])

    return f"This house is valued at ${prediction}"