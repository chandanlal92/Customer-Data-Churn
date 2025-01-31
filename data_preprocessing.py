
import data_preprocessing
from data_processing import preprocess_data
from data_processing import data


# Feature Engineering (if any specific feature creation is needed)
def feature_engineering(data):
    # Example: Create a new feature
    data['TenurePerMonth'] = data['tenure'] / data['TotalCharges']
    return data



data = feature_engineering(data)
