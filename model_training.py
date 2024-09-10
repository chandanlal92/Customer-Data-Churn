import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from data_processing import preprocess_data
from data_processing import data
from data_processing import X_test,X_train,y_test,y_train


# Train the model
def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

model = train_model(X_train, y_train)

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))

evaluate_model(model, X_test, y_test)

# Save the model
joblib.dump(model, 'churn_model.pkl')
