# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 12:46:16 2022

@author: culro
"""

import requests
import pandas as pd
import os
import time
from sklearn.metrics import confusion_matrix, classification_report
from statistics import mean
# import numpy as np
import json

os.chdir(os.path.dirname(os.path.abspath(__file__)))

x = pd.read_csv('Data\\x_test.csv')
y = pd.read_csv('Data\\y_test.csv')
# input_len = (len(x))
input_len = 10

x = x.to_numpy()[:input_len]
y = y.to_numpy()[:input_len]

y_new = []
y_new = [2 if o == 5 else 1 if o == 3 else 0 for o in y]

prediction = []
pred_time = []

# for i in range(len(y)):
#     y[i] = (int(y[i]))
#     if y[i] == 5:
#         y_new.append(2)
#     elif y[i] == 3:
#         y_new.append(1)
#     elif y[i] == 1:
#         y_new.append(0)


for i in x:
    i = str(i).replace("/","")    
    # url = 'http://127.0.0.1:8000/predict/'
    url = 'http://18.217.161.250/app/predict/'
    data =  {"review": str(i)}
    start = time.time()
    
    r_resp =requests.post(url, data = json.dumps(data))
    duration = time.time() - start
    pred_time.append(duration)
    r_dict = r_resp.json()
    pred = int(r_dict["rating"])
    print(pred)
    prediction.append(pred)
    
    
print(classification_report(y_new, prediction))
print("Avergae prediction time:", round(mean(pred_time),4), "seconds")
print("Total prediction time:", round(sum(pred_time),4), "seconds")


#%%
print(type(classification_report(y_new, prediction)))
report = classification_report(y_new, prediction, output_dict=True)
df = pd.DataFrame(report).transpose()
df["Avergae prediction time (seconds)"] = round(mean(pred_time),4)
df["Total prediction time (seconds)"] = round(sum(pred_time),4)
df.to_csv(str(input_len) + '_obs_test.csv')

df = pd.DataFrame(prediction)
df.to_csv(str(input_len) + '_y_pred.csv')
df.to_csv(str(input_len) + '_y_new.csv')



