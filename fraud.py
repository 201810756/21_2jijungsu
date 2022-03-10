"""
@author: 201810756 Kimdaehwan
@date: 20211108
@description: First question for Midterm
    Implement the Gaussian Naive Bayes model.
    Data: 'fraud_data.csv'
"""
# Import library of Gaussian Naive Bayes model

from sklearn.naive_bayes import GaussianNB
from sklearn import datasets

import pandas as pd
import numpy as np


# replace the dataset to suit model
def replace(df):
    df = df.replace(['paid', 'current', 'arrears'], [2, 1, 0])
    df = df.replace(['none', 'guarantor', 'coapplicant'], [0, 1, 1])
    df = df.replace(['coapplicant'], [1])
    df = df.replace(['rent', 'own'], [0, 1])
    #df["Fraud"]=df["Fraud"].astype(int)
    df = df.replace(['False', 'True'], [0,1])
    df = df.replace(['none'], [float('NaN')])
    df = df.replace(['free'], [-1])
    return df
df = pd.read_csv('./fraud_data.csv')
res = replace(df)

# training data
history = res["History"]
coapplicant = res["CoApplicant"]
accommodation = res["Accommodation"]
X=pd.Series(history,coapplicant,accommodation)
Y =list(res["Fraud"])

# training
model = GaussianNB()
model.fit(X, Y)
# test using the given case
predicted1 = model.predict([[2,0,0]])
pred_prob1 = model.predict_proba([[2,0,0]])
predicted2 = model.predict([[2,1,0]])
pred_prob2 = model.predict_proba([[2,1,0]])
predicted3 = model.predict([[0,1,0]])
pred_prob3 = model.predict_proba([[0,1,0]])
predicted4 = model.predict([[0,1,1]])
pred_prob4 = model.predict_proba([[0,1,1]])
predicted5 = model.predict([[0,1,1]])
pred_prob5 = model.predict_proba([[0,1,1]])

# result
print('paid/none/rent result:',predicted1,'prob:',pred_prob1)
print('paid/guarantor/rent result:',predicted2,'prob:',pred_prob2)
print('arrears/guarantor/rent result:',predicted3,'prob:',pred_prob3)
print('arrears/guarantor/own result:',predicted4,'prob:',pred_prob4)
print('arrears/coapplicant/own result:',predicted5,'prob:',pred_prob5)