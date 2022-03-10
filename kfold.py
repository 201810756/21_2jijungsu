"""
@author: 201810756 Kimdaehwan
@date: 20211108
@description: Second question for Midterm
    Implement the Gaussian Naive Bayes model(using K-fold cross validation)
    Data: 'Student_Network_Ads.csv'
"""

# Import library of Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold # for check
from sklearn import datasets

import pandas as pd
import numpy as np

# test accuracy
def test(data,predicted):
    return (np.sum(data==predicted))/len(data)*100
# replace the dataset to suit model
def replace(df):
    df = df.replace(['Male','Female'],[1,0])
    return df
df = pd.read_csv('./Social_Network_Ads.csv')
res = replace(df)

# check prediction accuracy by sklearn
"""i=0
kfold=KFold(n_splits=10)
precision=0
for train_index, test_index in kfold.split(res):
    i+=1
    training_data=res.iloc[train_index]
    testing_data=res.iloc[test_index]
    training_gender = training_data["Gender"]
    training_age = training_data["Age"]
    training_salary = training_data["EstimatedSalary"]
    training_X = list(zip(training_gender, training_age, training_salary))
    training_Y = list(training_data["Purchased"])
    test_gender = testing_data["Gender"]
    test_age = testing_data["Age"]
    test_salary = testing_data["EstimatedSalary"]
    test_X = list(zip(test_gender, test_age, test_salary))
    test_Y = list(testing_data["Purchased"])
    model=GaussianNB()
    model.fit(training_X,training_Y)
    predicted=model.predict(test_X)
    print(i,'test accuracy result=',test(test_Y,predicted))
    precision+=test(test_Y,predicted)
average=precision/10
print("Accuracy by k-fold cross validation(use sklearn) : ",average)"""

# check prediction accuracy by self-made function
def data_split(dataset,k,i,data_length):
    j=i*(data_length)
    if i == 0:
        testing_data = dataset.iloc[:j+data_length].reset_index(drop=True)
        training_data = dataset.iloc[j+data_length:].reset_index(drop=True)
    elif i == k-1:
        testing_data = dataset.iloc[j:].reset_index(drop=True)
        training_data = dataset.iloc[:j].reset_index(drop=True)
    else:
        testing_data = dataset.iloc[j:j + data_length].reset_index(drop=True)
        training_data_head = dataset.iloc[:j].reset_index(drop=True)
        training_data_tail = dataset.iloc[j + data_length:].reset_index(drop=True)
        training_data = pd.concat([training_data_head, training_data_tail])
    return training_data, testing_data

def k_fold(dataset,k):
    accuracy=0
    data_length=int(len(dataset)/k)
    for i in range(0,k):
        training_data=data_split(dataset,k,i,data_length)[0]
        testing_data=data_split(dataset,k,i,data_length)[1]
        training_gender = training_data["Gender"]
        training_age = training_data["Age"]
        training_salary = training_data["EstimatedSalary"]
        training_X = list(zip(training_gender, training_age, training_salary))
        training_Y = list(training_data["Purchased"])
        test_gender = testing_data["Gender"]
        test_age = testing_data["Age"]
        test_salary = testing_data["EstimatedSalary"]
        test_X = list(zip(test_gender, test_age, test_salary))
        test_Y = list(testing_data["Purchased"])
        model = GaussianNB()
        model.fit(training_X, training_Y)
        predicted = model.predict(test_X)
        print((i+1), 'test accuracy result=', test(test_Y, predicted),'%')
        accuracy += test(test_Y, predicted)
    average_accuracy=accuracy/k
    print("Accuracy by k-fold cross validation(sklearn is not used) :",average_accuracy,'%')
    #print("same accuracy ? : ", average_accuracy==average)

k_fold(res,10)