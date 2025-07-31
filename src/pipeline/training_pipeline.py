from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainingPipeline:
    def main(self):
        """
        Runs the complete training pipeline.
        """
        try:
            # --- IMPORTANT: Replace with your S3 details ---
            S3_BUCKET_NAME = "dhanush-house-price-dataset-2025"
            S3_FILE_KEY = "train.csv"

            # Step 1: Data Ingestion
            ingestion = DataIngestion()
            raw_data_path = ingestion.initiate_data_ingestion(
                bucket_name=S3_BUCKET_NAME,
                key=S3_FILE_KEY
            )

            # Step 2: Data Transformation
            transformation = DataTransformation()
            transformed_data, _ = transformation.initiate_data_transformation(raw_data_path)

            # Step 3: Model Training
            trainer = ModelTrainer()
            trainer.initiate_model_training(transformed_data)

            print("--- Training Pipeline Completed Successfully ---")

        except Exception as e:
            print(f"An error occurred in the training pipeline: {e}")
            raise e

if __name__ == '__main__':
    pipeline = TrainingPipeline()
    pipeline.main()