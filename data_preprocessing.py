
# data_preprocessing.py
# This script is responsible for preprocessing the dataset and preparing it for model training.
import pandas as pd
import numpy as np
from data_processing import preprocess_data
from data_processing import data


def preprocess_data(data):
    # Handle missing values
    data.fillna(0, inplace=True) 
    # Convert categorical variables to numerical
    data = pd.get_dummies(data, drop_first=True)
    # Normalize or scale features if necessary
    # Example: from sklearn.preprocessing import StandardScaler
    # scaler = StandardScaler()
    # data[['feature1', 'feature2']] = scaler.fit_transform(data[['feature1', 'feature2']])
    # Ensure 'Churn' is treated as a binary classification target
    data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0}).astype(int)

    return data

# Feature Engineering (if any specific feature creation is needed)
def feature_engineering(data):
    # Example: Create a new feature
    data['TenurePerMonth'] = data['tenure'] / data['TotalCharges']
    # You can add more feature engineering steps here
    # Ensure the data is in the correct format
    data = preprocess_data(data)
    # Return the processed data     
    return data     



data = feature_engineering(data)
