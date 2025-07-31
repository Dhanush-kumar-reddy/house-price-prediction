# 🚀 End-to-End House Price Prediction Project

This project demonstrates a complete, industrial-standard MLOps pipeline for predicting house prices. It includes cloud data ingestion, a modular ML training pipeline, automated CI/CD with GitHub Actions, and deployment as a REST API.

---

### 🛠️ Tech Stack

- **Cloud:** AWS (S3 for storage, ECR for container registry)
- **Data Processing:** Pandas, NumPy, Scikit-learn
- **Modeling:** XGBoost, Scikit-learn
- **API:** FastAPI, Uvicorn
- **Deployment:** Docker
- **CI/CD:** GitHub Actions
- **Infrastructure as Code (IaC):** AWS CLI

---

### 📂 Project Structure

The project follows a modular and scalable structure:
- **`src/`**: Contains all source code, organized into components, pipelines, and utilities.
- **`notebooks/`**: Holds Jupyter notebooks for exploratory data analysis (EDA).
- **`app.py`**: The FastAPI application for serving predictions.
- **`.github/workflows/`**: Defines the CI/CD pipeline using GitHub Actions.
- **`Dockerfile`**: Defines the container for the application.

---

### ⚙️ How to Run

1.  **Set up the environment:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run the training pipeline:**
    ```bash
    python src/pipeline/training_pipeline.py
    ```
3.  **Run the FastAPI server locally:**
    ```bash
    uvicorn app:app --reload
    ```

---