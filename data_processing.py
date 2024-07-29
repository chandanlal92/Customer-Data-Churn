import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the dataset
data = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Preprocess the data
def preprocess_data(data):
    # Drop unnecessary columns
    data.drop(['customerID'], axis=1, inplace=True)
    
    # Handle missing values
    data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
    data.fillna(data.mean(), inplace=True)
    
    # Encode categorical variables
    le = LabelEncoder()
    for column in data.select_dtypes(include=['object']).columns:
        data[column] = le.fit_transform(data[column])
    
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
