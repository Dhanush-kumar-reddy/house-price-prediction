from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from src.pipeline.prediction_pipeline import PredictionPipeline

# Create the FastAPI app
app = FastAPI()

# Define the input data model using Pydantic
# All feature fields from the training data are included.
class HouseData(BaseModel):
    MSSubClass: int = Field(..., example=60)
    MSZoning: str = Field(..., example="RL")
    LotFrontage: float = Field(..., example=65.0)
    LotArea: int = Field(..., example=8450)
    Street: str = Field(..., example="Pave")
    LotShape: str = Field(..., example="Reg")
    LandContour: str = Field(..., example="Lvl")
    Utilities: str = Field(..., example="AllPub")
    LotConfig: str = Field(..., example="Inside")
    LandSlope: str = Field(..., example="Gtl")
    Neighborhood: str = Field(..., example="CollgCr")
    Condition1: str = Field(..., example="Norm")
    Condition2: str = Field(..., example="Norm")
    BldgType: str = Field(..., example="1Fam")
    HouseStyle: str = Field(..., example="2Story")
    OverallQual: int = Field(..., example=7)
    OverallCond: int = Field(..., example=5)
    YearBuilt: int = Field(..., example=2003)
    YearRemodAdd: int = Field(..., example=2003)
    RoofStyle: str = Field(..., example="Gable")
    RoofMatl: str = Field(..., example="CompShg")
    Exterior1st: str = Field(..., example="VinylSd")
    Exterior2nd: str = Field(..., example="VinylSd")
    MasVnrType: str = Field(..., example="BrkFace")
    MasVnrArea: float = Field(..., example=196.0)
    ExterQual: str = Field(..., example="Gd")
    ExterCond: str = Field(..., example="TA")
    Foundation: str = Field(..., example="PConc")
    BsmtQual: str = Field(..., example="Gd")
    BsmtCond: str = Field(..., example="TA")
    BsmtExposure: str = Field(..., example="No")
    BsmtFinType1: str = Field(..., example="GLQ")
    BsmtFinSF1: int = Field(..., example=706)
    BsmtFinType2: str = Field(..., example="Unf")
    BsmtFinSF2: int = Field(..., example=0)
    BsmtUnfSF: int = Field(..., example=150)
    TotalBsmtSF: int = Field(..., example=856)
    Heating: str = Field(..., example="GasA")
    HeatingQC: str = Field(..., example="Ex")
    CentralAir: str = Field(..., example="Y")
    Electrical: str = Field(..., example="SBrkr")
    FirstFlrSF: int = Field(..., alias="1stFlrSF", example=856)
    SecondFlrSF: int = Field(..., alias="2ndFlrSF", example=854)
    LowQualFinSF: int = Field(..., example=0)
    GrLivArea: int = Field(..., example=1710)
    BsmtFullBath: int = Field(..., example=1)
    BsmtHalfBath: int = Field(..., example=0)
    FullBath: int = Field(..., example=2)
    HalfBath: int = Field(..., example=1)
    BedroomAbvGr: int = Field(..., example=3)
    KitchenAbvGr: int = Field(..., example=1)
    KitchenQual: str = Field(..., example="Gd")
    TotRmsAbvGrd: int = Field(..., example=8)
    Functional: str = Field(..., example="Typ")
    Fireplaces: int = Field(..., example=0)
    FireplaceQu: str = Field(..., example="NA")
    GarageType: str = Field(..., example="Attchd")
    GarageYrBlt: float = Field(..., example=2003.0)
    GarageFinish: str = Field(..., example="RFn")
    GarageCars: int = Field(..., example=2)
    GarageArea: int = Field(..., example=548)
    GarageQual: str = Field(..., example="TA")
    GarageCond: str = Field(..., example="TA")
    PavedDrive: str = Field(..., example="Y")
    WoodDeckSF: int = Field(..., example=0)
    OpenPorchSF: int = Field(..., example=61)
    EnclosedPorch: int = Field(..., example=0)
    ThreeSsnPorch: int = Field(..., alias="3SsnPorch", example=0)
    ScreenPorch: int = Field(..., example=0)
    PoolArea: int = Field(..., example=0)
    MiscVal: int = Field(..., example=0)
    MoSold: int = Field(..., example=2)
    YrSold: int = Field(..., example=2008)
    SaleType: str = Field(..., example="WD")
    SaleCondition: str = Field(..., example="Normal")

    class Config:
        populate_by_name = True

@app.get("/")
def read_root():
    return {"message": "Welcome to the House Price Prediction API"}

@app.post("/predict")
def predict_price(data: HouseData):
    """Endpoint to predict house price."""
    try:
        # Convert the input data to a pandas DataFrame
        # The alias handling is done automatically by Pydantic's model_dump
        input_df = pd.DataFrame([data.model_dump(by_alias=True)])
        
        # Create an instance of the prediction pipeline and make a prediction
        pipeline = PredictionPipeline()
        prediction = pipeline.predict(input_df)
        
        return {"predicted_price": round(prediction, 2)}

    except Exception as e:
        return {"error": str(e)}