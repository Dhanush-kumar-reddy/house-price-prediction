import os
from dataclasses import dataclass

# Models
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
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

    def evaluate_models(self, X_train, y_train, X_test, y_test, models):
        """
        Trains and evaluates a dictionary of models, returning a report.
        """
        report = {}
        for model_name, model in models.items():
            model.fit(X_train, y_train) # Train model
            y_test_pred = model.predict(X_test)
            test_model_score = r2_score(y_test, y_test_pred)
            report[model_name] = test_model_score
        return report

    def initiate_model_training(self, transformed_array):
        """
        Identifies the best model and saves it.
        """
        try:
            print("Splitting data into training and testing sets.")
            X = transformed_array[:, :-1]
            y = transformed_array[:, -1]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Dictionary of models to evaluate
            models = {
                "Linear Regression": LinearRegression(),
                "Random Forest": RandomForestRegressor(random_state=42),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "XGBoost": XGBRegressor(random_state=42)
            }
            
            # Get model performance report
            model_report:dict = self.evaluate_models(X_train, y_train, X_test, y_test, models)
            
            # Find the best model score and name from the report
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            print(f"\n--- Model Evaluation Report ---")
            print(model_report)
            print(f"===================================")
            print(f"Best Model Found: {best_model_name} with R2 Score: {best_model_score:.4f}")
            print(f"===================================")
            
            # Save the best performing model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            print(f"Best model ({best_model_name}) saved to {self.model_trainer_config.trained_model_file_path}")

        except Exception as e:
            raise e
        