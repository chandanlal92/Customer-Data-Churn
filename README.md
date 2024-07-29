# Customer Churn Prediction

This project aims to predict customer churn using machine learning techniques.

## Project Structure
1. **Data Collection and Preprocessing**
2. **Exploratory Data Analysis (EDA)**
3. **Feature Engineering**
4. **Model Training and Evaluation**
5. **Model Deployment**

## Installation

1. Clone the repository
2. Install the required packages:
   ```sh
   pip install -r requirements.txt
3. Run the application:
   ```sh
   python app.py
4. Usage
   1. To train the model, run:
   ```sh
   python model_training.py

2. To start the Flask API:
   ```sh
   python app.py

3. Make predictions by sending a POST request to http://127.0.0.1:5000/predict with JSON data:
{
   "data": [[1, 0, 45, ...]]
}

