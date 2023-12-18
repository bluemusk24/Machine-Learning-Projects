
import pickle

model_file = 'model_C=10.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)


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


X = dv.transform(patient)
y_pred = model.predict_proba(X)[0,1]         


print('input', patient)
print('diabetes prediction', y_pred)

if y_pred >= 0.5:
    print('start seeking medical attention to get insulin for diabetes treatment')
else:
    print('patient has no sign of early-stage diabetes risk')




patient_1 = {
    "gender": "female",
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


X = dv.transform(patient_1)
y_pred = model.predict_proba(X)[0,1] 

print('input', patient)
print('diabetes prediction', y_pred)

if y_pred >= 0.5:
    print('start seeking medical attention to get insulin for diabetes treatment')
else:
    print('patient has no sign of early-stage diabetes risk')





