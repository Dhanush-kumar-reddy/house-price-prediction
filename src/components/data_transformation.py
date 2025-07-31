import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from src.utils.common import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', "preprocessor.joblib")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self, raw_data_path):
        """
        This method applies all the data cleaning and transformation steps
        discovered during the EDA.
        """
        try:
            # 1. Load the raw data
            df = pd.read_csv(raw_data_path)

            # 2. Drop the 'Id' column as it's not a feature
            df.drop('Id', axis=1, inplace=True)

            # 3. Log transform the target variable 'SalePrice' to handle skewness
            df['SalePrice'] = np.log1p(df['SalePrice'])

            # 4. Separate features (X) and target (y)
            X = df.drop(columns='SalePrice', axis=1)
            y = df['SalePrice']

            # 5. Define numerical and categorical features
            numerical_features = X.select_dtypes(include=np.number).columns
            categorical_features = X.select_dtypes(include=object).columns

            # 6. Create preprocessing pipelines for numerical and categorical data
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')), # Fill missing with median
                ('scaler', StandardScaler())                   # Scale features
            ])

            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')), # Fill missing with mode
                ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)), # One-hot encode
                ('scaler', StandardScaler(with_mean=False))           # Scale encoded features
            ])

            # 7. Combine pipelines into a single ColumnTransformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_features),
                    ("cat_pipeline", cat_pipeline, categorical_features)
                ]
            )

            # 8. Apply the full preprocessing pipeline to the feature set
            X_processed = preprocessor.fit_transform(X)
            
            # 9. Combine processed features and the target variable
            transformed_arr = np.c_[X_processed, np.array(y)]

            # 10. Save the preprocessor object for future use
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor
            )
            print("Preprocessor object saved successfully.")

            return transformed_arr, self.data_transformation_config.preprocessor_obj_file_path
            
        except Exception as e:
            raise e