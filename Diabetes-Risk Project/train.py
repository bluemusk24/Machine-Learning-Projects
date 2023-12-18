#!/usr/bin/env python
# coding: utf-8


import pickle

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# Parameters

C = 10
n_splits = 5 
output_file = f'model_C={C}.bin'


# data preparation

df = pd.read_csv("C:\\Users\\emman\\ML-deployment\\Captsone project\\diabetes_risk_prediction_dataset.csv")

df.columns = df.columns.str.lower().str.replace(' ','_')

categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower()

df['class'] = (df['class'] == 'positive').astype(int)

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

categorical_variables = ['gender', 'polyuria', 'polydipsia', 'sudden_weight_loss','weakness', 'polyphagia', 'genital_thrush', 'visual_blurring',
                         'itching', 'irritability', 'delayed_healing', 'partial_paresis','muscle_stiffness', 'alopecia', 'obesity']

numerical_variables = ['age']


# Training function

def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical_variables + numerical_variables].to_dict(orient='records')
    
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)
    
    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)
    
    return dv, model


# Prediction function:

def predict(df, dv, model):
    dicts = df[categorical_variables + numerical_variables].to_dict(orient='records')
    
    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:,1]
    
    return y_pred


# validation

print(f'doing validation with C={C}')

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

scores = []

fold = 0 

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]
    
    y_train = df_train['class'].values
    y_val = df_val['class'].values
    
    dv, model = train(df_train, y_train, C=C)
    y_pred = predict(df_val, dv, model)
    
    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

    print(f'auc on fold {fold} is {auc}')
    fold = fold + 1

print('validation results:')  
print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores))) 


# train the final model

print('training of the final model')

dv, model = train(df_full_train, df_full_train['class'].values, C=10)
y_pred = predict(df_test, dv, model)

y_test = df_test['class'].values
auc = roc_auc_score(y_test, y_pred)

print(f'auc={auc}')


# Save the Model as File

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f'the model is saved to this {output_file}')
