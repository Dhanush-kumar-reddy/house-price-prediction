import joblib
import pandas as pd
import numpy as np

class PredictionPipeline:
    def predict(self, features):
        """
        Loads the preprocessor and model to make a prediction.

        Args:
            features (pd.DataFrame): DataFrame with the input features.

        Returns:
            float: The predicted house price.
        """
        try:
            # Load the saved preprocessor and model objects
            preprocessor = joblib.load('artifacts/preprocessor.joblib')
            model = joblib.load('artifacts/model.joblib')

            # Apply the preprocessing steps
            data_scaled = preprocessor.transform(features)

            # Make a prediction
            prediction = model.predict(data_scaled)

            # Inverse transform the prediction (since we log-transformed the target)
            return np.expm1(prediction[0])

        except Exception as e:
            raise e