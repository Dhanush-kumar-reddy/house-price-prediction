import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.utils.common import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', "preprocessor.joblib")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """This function is responsible for creating the data transformation pipeline."""
        try:
            # Define which columns should be scaled and which should be one-hot encoded
            numerical_columns = ["LotFrontage", "LotArea", "OverallQual", "OverallCond", "YearBuilt"] # Add more as needed
            categorical_columns = ["MSZoning", "Street", "LotShape", "LandContour", "Utilities"] # Add more as needed

            # Create pipelines for numerical and categorical data
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")), # Handle missing values
                ("scaler", StandardScaler()) # Scale the data
            ])
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ("scaler", StandardScaler(with_mean=False))
            ])

            # Combine pipelines into a single preprocessor object
            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ])
            return preprocessor
        except Exception as e:
            raise e

    def initiate_data_transformation(self, raw_data_path):
        """Applies the transformation pipeline to the raw data."""
        try:
            df = pd.read_csv(raw_data_path)
            preprocessor_obj = self.get_data_transformer_object()

            # For this example, we'll use a subset of columns for simplicity
            target_column_name = "SalePrice"
            all_feature_columns = [col for col in df.columns if col != target_column_name]

            input_feature_df = df.drop(columns=[target_column_name], axis=1)
            target_feature_df = df[target_column_name]

            # Apply the preprocessor to the feature data
            input_feature_arr = preprocessor_obj.fit_transform(input_feature_df)

            # Combine processed features and the target variable
            transformed_arr = np.c_[input_feature_arr, np.array(target_feature_df)]

            # Save the preprocessor object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            print("Preprocessor object saved.")

            return transformed_arr, self.data_transformation_config.preprocessor_obj_file_path
        except Exception as e:
            raise e

# Example of how to run this component
if __name__ == "__main__":
    # This assumes you have already run the data ingestion step
    raw_data_path = 'artifacts/data.csv'

    transformation = DataTransformation()
    transformed_data, _ = transformation.initiate_data_transformation(raw_data_path)
    # You can now proceed to split this data into train/test sets for model training