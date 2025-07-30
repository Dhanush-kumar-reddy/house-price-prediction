import os
from dataclasses import dataclass
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from src.utils.common import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.joblib")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, transformed_array):
        """
        Trains a model on the transformed data and saves it.
        """
        try:
            print("Splitting data into training and testing sets.")
            # The last column is the target variable (SalePrice)
            X = transformed_array[:, :-1]
            y = transformed_array[:, -1]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Initialize and train the XGBoost Regressor model
            print("Training XGBoost Regressor model.")
            model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
            model.fit(X_train, y_train)

            # Make predictions and evaluate the model
            y_pred = model.predict(X_test)
            score = r2_score(y_test, y_pred)
            print(f"Model evaluation complete. R2 Score: {score:.4f}")

            # Save the trained model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )
            print(f"Model saved to {self.model_trainer_config.trained_model_file_path}")

        except Exception as e:
            raise e

# Example of how to run this component (for testing purposes)
if __name__ == '__main__':
    # This part assumes you have already run the previous two components
    # to get the transformed_data array. For a full pipeline, we'll
    # orchestrate these calls differently.

    # This is a placeholder for the transformed data from the previous step
    # In a real run, this would be the output of DataTransformation
    from src.components.data_ingestion import DataIngestion
    from src.components.data_transformation import DataTransformation

    # Note: You would need to provide your S3 details here to run this standalone
    # S3_BUCKET_NAME = "your-unique-bucket-name"
    # S3_FILE_KEY = "train.csv"

    # ingestion = DataIngestion()
    # raw_data_path = ingestion.initiate_data_ingestion(S3_BUCKET_NAME, S3_FILE_KEY)

    # transformation = DataTransformation()
    # transformed_data, _ = transformation.initiate_data_transformation(raw_data_path)

    # trainer = ModelTrainer()
    # trainer.initiate_model_training(transformed_data)
    pass # We will create a separate pipeline script to run this.