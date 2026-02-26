import pickle

# Globals
C = 1.0
input_file = f"model_C={C}.bin"

# Load the model
# Returns the tuple as it was saved
print(f"Loading model from {input_file}")
with open(input_file, "rb") as f_in:
    dv, model = pickle.load(f_in)

# Sample customer
customer = {
    "gender": "female",
    "seniorcitizen": 0,
    "partner": "yes",
    "dependents": "no",
    "phoneservice": "no",
    "multiplelines": "no_phone_service",
    "internetservice": "dsl",
    "onlinesecurity": "no",
    "onlinebackup": "yes",
    "deviceprotection": "no",
    "techsupport": "no",
    "streamingtv": "no",
    "streamingmovies": "no",
    "contract": "month-to-month",
    "paperlessbilling": "yes",
    "paymentmethod": "electronic_check",
    "tenure": 1,
    "monthlycharges": 29.85,
    "totalcharges": 29.85,
}

# Create a feature matrix based on customer
print(f"Predicting customer:\n {'\n '.join(f'{k}: {v}' for k, v in customer.items())}")
X = dv.transform([customer])
y_pred = model.predict_proba(X)  # 38% to churn
print(f"Churn rate for prediction is {y_pred[0][1]:.2%}")
