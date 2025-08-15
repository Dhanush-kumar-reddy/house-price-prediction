import joblib
import pandas as pd
import numpy as np

class PredictionPipeline:
    def predict(self, features):
        """
        Loads artifacts and makes a prediction on new data.
        This method now ensures the input data structure matches the training data.

        Args:
            features (pd.DataFrame): DataFrame with the raw input features from the API.

        Returns:
            float: The predicted house price.
        """
        try:
            # Load the saved preprocessor, model, and training columns
            preprocessor = joblib.load('artifacts/preprocessor.joblib')
            model = joblib.load('artifacts/model.joblib')
            training_columns = joblib.load('artifacts/training_columns.joblib')

            # --- THE FIX ---
            # Reindex the input DataFrame to match the columns the model was trained on.
            # This adds any missing columns (and fills them with 0) and ensures the order is identical.
            features = features.reindex(columns=training_columns, fill_value=0)

            # Apply the preprocessing steps
            data_scaled = preprocessor.transform(features)

            # Make a prediction
            prediction = model.predict(data_scaled)

            # Inverse transform the prediction (since we log-transformed the target)
            return np.expm1(prediction[0])

        except Exception as e:
            raise e