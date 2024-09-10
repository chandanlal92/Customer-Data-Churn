import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import numpy as np

# Load the dataset
data = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Preprocess the data
def preprocess_data(data):
    # Make a copy of the dataset to avoid modifying the original DataFrame
    data = data.copy()

    # Drop unnecessary columns
    data.drop(['customerID'], axis=1, inplace=True)

    # Convert 'TotalCharges' to numeric, and coerce errors into NaN
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

    # Fill missing 'TotalCharges' values using .loc[] to avoid chained assignment
    data.loc[data['TotalCharges'].isna(), 'TotalCharges'] = data['TotalCharges'].median()

     # Ensure 'tenure' and other numeric columns are actually numeric
    data['tenure'] = pd.to_numeric(data['tenure'], errors='coerce')


    # Fill missing 'tenure' values if any (though there shouldn't be any in this dataset)
    data.loc[data['tenure'].isna(), 'tenure'] = data['tenure'].median()

    # Encode categorical variables using .loc[] to avoid chained assignment
    le = LabelEncoder()
    for column in data.select_dtypes(include=['object']).columns:
        data.loc[:, column] = le.fit_transform(data[column])

    # **Ensure 'Churn' is treated as a binary classification target**
    # Check if 'Churn' column is 'Yes'/'No' and convert to 1/0
    data['Churn'] = data['Churn'].map({1: 1, 0: 0}).astype(int)
    # Split data into features and target
    X = data.drop(['Churn'], axis=1)
    y = data['Churn']

    # Normalize/standardize numerical features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = preprocess_data(data)
