# Load the Model

import pickle

from flask import Flask
from flask import request
from flask import jsonify


model_file = 'model_C=10.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('diabetes_risk')

@app.route('/predict', methods=['POST'])
def predict():

    patient = request.get_json()

    X = dv.transform([patient])
    y_pred = model.predict_proba(X)[0,1]
    diabetes_risk = y_pred >= 0.5

    result = {
        'diabetes_risk_probability': float(y_pred),
        'diabetes_risk': bool(diabetes_risk)
    }

    return jsonify(result)


if __name__== "__main__": 
    app.run(debug=True, host='localhost', port=9696)
