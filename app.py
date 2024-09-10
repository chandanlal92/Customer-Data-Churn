from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('churn_model.pkl')


@app.route('/')
def home():
    return "Welcome to the Machine Learning API!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    predict_request = np.array(data['data'])
    prediction = model.predict(predict_request)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
