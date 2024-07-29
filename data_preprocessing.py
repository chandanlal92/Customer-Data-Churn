# Feature Engineering (if any specific feature creation is needed)
def feature_engineering(data):
    # Example: Create a new feature
    data['TenurePerMonth'] = data['tenure'] / data['TotalCharges']
    return data

data = feature_engineering(data)
X_train, X_test, y_train, y_test = preprocess_data(data)
