"""
@author: 201810756 Kimdaehwan
@date: 20211108
@description: Second question for Midterm
    Implement the Gaussian Naive Bayes model(using Hold-out)
    Data: 'Student_Network_Ads.csv'
"""
# Import library of Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB
from sklearn import datasets
import pandas as pd
import numpy as np
# for ROC curve & AUC score
from sklearn.metrics import roc_curve,roc_auc_score
import matplotlib.pyplot as plt


# test accruacy
def test_accuracy(target,predicted):
    accuracy=np.sum(target==predicted)/len(target)*100
    print('The prediction accuracy(by Hold-out) : ',accuracy,'%')

# replace the dataset to suit model
def replace(df):
    df = df.replace(['Male','Female'],[1,0])
    return df
df = pd.read_csv('./Social_Network_Ads.csv')
res = replace(df)

# split data for Hold-out learning
def train_test_split(dataset):
    training_data = dataset.iloc[:320].reset_index(drop=True)
    testing_data=dataset.iloc[320:].reset_index(drop=True)
    return training_data,testing_data
training_data = train_test_split(res)[0]
testing_data = train_test_split(res)[1]

# making training dataset
training_gender=training_data["Gender"]
training_age=training_data["Age"]
training_salary=training_data["EstimatedSalary"]
training_X=list(zip(training_gender,training_age,training_salary))
training_Y=list(training_data["Purchased"])

# making test dataset
test_gender=testing_data["Gender"]
test_age=testing_data["Age"]
test_salary=testing_data["EstimatedSalary"]
test_X=list(zip(test_gender,test_age,test_salary))
test_Y=list(testing_data["Purchased"])


model=GaussianNB()
# learning training dataset
model.fit(training_X,training_Y)
# predict using test dataset
predicted=model.predict(test_X)
# show accuracy
test_accuracy(test_Y,predicted)
# TP,FN,FP,TN for precision, recall, sensitivity, specificity
def confusion_matrix(test_data,predict_data):
    target=pd.Series(test_data)
    predict=pd.Series(predict_data)
    confusion_m=pd.crosstab(target,predict,rownames=['target'],colnames=['predict'])
    print('Confusion Matrix')
    print(confusion_m)
    confusion_m2=confusion_m.reindex(columns=[1,0],index=[1,0])
    print('Confusion Matrix(Modified)')
    print(confusion_m2)
    TN = confusion_m.iloc[0,0]
    FP = confusion_m.iloc[0,1]
    FN = confusion_m.iloc[1,0]
    TP = confusion_m.iloc[1,1]
    print('TN:',TN,'FP:',FP,'FN:',FN,'TP:',TP)
    return TN,FP,FN,TP
# precision, recall, sensitivity, specificity
def evaluation(tn,fp,fn,tp):
    precision=tp/(tp+fp)
    recall=tp/(fn+tp)
    specificity=tn/(fp+tn)
    return precision,recall,specificity
tn,fp,fn,tp=confusion_matrix(test_Y,predicted)
print('Precision score : ', evaluation(tn,fp,fn,tp)[0])
print('Recall / Sensitivity score : ', evaluation(tn,fp,fn,tp)[1])
print('Specificity score : ', evaluation(tn,fp,fn,tp)[2])

# Check Auc score and make ROC curve by Sklearn
def show_roc_curve(FPR,TPR):
    plt.plot(FPR,TPR,color='red',label='ROC')
    plt.plot([0,1],[0,1],color='blue',linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve(Hold-out)')
    plt.legend()
    plt.show()
FPR,TPR,thresholds=roc_curve(test_Y,predicted)
auc_score=roc_auc_score(test_Y,predicted)
print('AUC_score(by sklearn):',auc_score)
show_roc_curve(FPR,TPR)