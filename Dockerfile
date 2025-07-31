# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's code into the container
# This includes the 'src' directory, 'app.py', and 'artifacts'
COPY . .

# Expose the port that FastAPI runs on
EXPOSE 8000

# Command to run the app when the container launches
# It will run the 'app' object from the 'app.py' file
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]