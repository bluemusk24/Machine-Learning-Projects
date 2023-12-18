#!/usr/bin/env python
# coding: utf-8

import requests

#url = 'http://localhost:9696/predict'

#host = 'diabetes-risk-serving-env.eba-pqmuv98q.us-east-1.elasticbeanstalk.com'
#url = 'http://{host}/predict'

url = 'http://aaf59f3c33cdc4ccb970de4fd0f68401-348694010.us-east-1.elb.amazonaws.com/predict'     # Loadbalancer URL


patient = {                                                # ensure double quote in json format
    "gender": "male",
    "polyuria": "no",
    "polydipsia": "no",
    "sudden_weight_loss": "no",
    "weakness": "yes",
    "polyphagia": "no",
    "genital_thrush": "yes",
    "visual_blurring": "no",
    "itching": "yes",
    "irritability": "no",
    "delayed_healing": "yes",
    "partial_paresis": "no",
    "muscle_stiffness": "no",
    "alopecia": "yes",
    "obesity": "no",
    "age": 61}



requests.post(url, json=patient)           #we send the patient as a post request to get a 200 response

response = requests.post(url, json=patient).json()
print(response)                                               # .json() gives outcome to python dictionary


if response['diabetes_risk'] == True:
    print('start seeking medical attention to get insulin for diabetes treatment')
else:
    print('patient has no symptoms of early-stage diabetes risk')


# #### Testing another patient


patient_1 = {
    "gender": "female",
    "polyuria": "yes",
    "polydipsia": "yes",
    "sudden_weight_loss": "yes",
    "weakness": "no",
    "polyphagia": "yes",
    "genital_thrush": "no",
    "visual_blurring": "yes",
    "itching": "no",
    "irritability": "yes",
    "delayed_healing": "no",
    "partial_paresis": "yes",
    "muscle_stiffness": "yes",
    "alopecia": "no",
    "obesity": "yes",
    "age": 30
}


requests.post(url, json=patient_1) 


response = requests.post(url, json=patient_1).json()
print(response)


if response['diabetes_risk'] == True:
    print('start seeking medical attention to get insulin for diabetes treatment')
else:
    print('patient has no symptoms of early-stage diabetes risk')

