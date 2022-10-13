#!/usr/bin/env python
# coding: utf-8

import pickle
from flask import Flask, request, jsonify

model_file = 'model2.bin'
dv_file = 'dv.bin'

with open(model_file,'rb') as f_model:
    model = pickle.load(f_model)

with open(dv_file,'rb') as f_dv:
    dv = pickle.load(f_dv)

app = Flask('credit_card')

@app.route('/predict', methods=['POST'])
def predict():
    client = request.get_json()
    X = dv.transform([client])
    y_prob_pred = model.predict_proba(X)[0,1]
    card = (y_prob_pred >= 0.5)
    result = {'card_probability': float(y_prob_pred), 'card':bool(card)}
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)


