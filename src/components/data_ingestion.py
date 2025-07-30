import os
import sys
from dataclasses import dataclass
import pandas as pd
import boto3
from botocore.exceptions import NoCredentialsError

# Assuming your custom logger and exception are set up in these paths
# from src.logger import logging
# from src.exception import CustomException

@dataclass
class DataIngestionConfig:
    # We define paths where the data will be stored after ingestion
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self, bucket_name: str, key: str):
        """
        Method to ingest data from an S3 bucket.

        Args:
            bucket_name (str): The name of the S3 bucket.
            key (str): The file key (path) within the S3 bucket.
        """
        print("Data ingestion process started.")
        # logging.info("Data ingestion process started.")

        try:
            # Create the artifacts directory if it doesn't exist
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            # Use boto3 to download the file from S3
            s3 = boto3.client('s3')
            print(f"Downloading data from S3 bucket '{bucket_name}' with key '{key}'...")
            s3.download_file(bucket_name, key, self.ingestion_config.raw_data_path)
            print(f"Data downloaded successfully and saved to {self.ingestion_config.raw_data_path}")

            return self.ingestion_config.raw_data_path

        except NoCredentialsError:
            error_message = "AWS credentials not found. Configure your credentials (e.g., via AWS CLI)."
            print(f"ERROR: {error_message}")
            # logging.error(error_message)
            # raise CustomException(error_message, sys)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            # logging.error(f"An unexpected error occurred: {e}")
            # raise CustomException(e, sys)

# Example of how to run this component
if __name__ == "__main__":
    # --- IMPORTANT ---
    # Before running, make sure your AWS credentials are configured.
    # The easiest way is to install the AWS CLI and run `aws configure`.

    # Replace with your bucket name and file key
    S3_BUCKET_NAME = "dhanush-house-price-dataset-2025" 
    S3_FILE_KEY = "train.csv"

    ingestion = DataIngestion()
    data_path = ingestion.initiate_data_ingestion(bucket_name=S3_BUCKET_NAME, key=S3_FILE_KEY)