"""
@author : 201810756 kimdaehwan
@date : 2021.11
@description : Nearest Neighbor Algorithm by euclidean distance(k=3)
"""
# import pandas, numpy, voronoi
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
import pandas as pd
import math as mt
import numpy as np
draft=pd.read_csv('../기말고사 대체과제_1/Nearest/draft.csv')

# euclidean distance
def euclidean(train,test):
    distance=0.0
    for i in range(len(train)):
        distance+=(train[i]-test[i])**2
    return mt.sqrt(distance)
# k-nearest neighbor
#def knn(training,target,test,k):
result = []
def knn(dataset,test,k):
    training = dataset.iloc[:, [1, 2]].values.tolist()
    distance=[0]*len(training)
    for i in range(len(training)):
        distance[i]=euclidean(training[i],test)
    dis=pd.DataFrame(distance,columns=['Distance'])
    data=((pd.concat([dataset,dis],axis=1)).sort_values('Distance',ascending=True)).reset_index(drop=True)
    labels,counts=np.unique(data.loc[0:k-1,['Draft']],return_counts=True)
    maxindex = 0
    max=0
    for i in range(len(counts)):
        if counts[i]>max:
            maxindex=i
            max=counts[i]
    result.append(labels[maxindex])
    #print(test,'result label:',result)
    #draft.loc[len(draft)]=[(len(draft)+1),test[0],test[1],result]
"""
print('-------3NN by euclidean--------')
knn(draft,[6.75,3.0],3)
knn(draft,[5.34,6.0],3)
knn(draft,[4.67,8.4],3)
knn(draft,[7.0,7.0],3)
knn(draft,[7.8,5.4],3)

"""
"""points=draft[['Speed','Agility']]
vor=Voronoi(points)
fig=voronoi_plot_2d(vor,point_size=1)
non_draft=draft[draft['Draft']=='No']
draft=draft[draft['Draft']=='Yes']
non_draft_sc=plt.scatter(non_draft['Speed'],non_draft['Agility'],marker='x',color='b',s=50)
draft_sc=plt.scatter(draft['Speed'],draft['Agility'],marker='o',color='r',s=50)
plt.legend((non_draft_sc,draft_sc),('non-draft','draft'),loc='upper left')
plt.title("3nn by euclidean")
plt.xlabel("Speed")
"""#plt.ylabel("Agility")
#plt.show()
"""
test data 
(6.75,3.0)
(5.34,6.0)
(4.67,8.4)
(7.0,7.0)
(7.8,5.4)
순서대로 추가됨 
"""
"""def replace(df):
    df=df.replace(['No','Yes'],[0,1])
    return df
aa=replace(draft)
"""
def train_test_split(dataset):
    training_data=dataset.iloc[:10].reset_index(drop=True)
    testing_data=dataset.iloc[10:].reset_index(drop=True)
    return training_data,testing_data
training_data=train_test_split(draft)[0]
testing_data=train_test_split(draft)[1]
testing= draft.iloc[:, [1, 2]].values.tolist()
testing_result=testing_data['Draft'].values.tolist()
print(testing_result)
for k in range(len(testing_result)):
    knn(training_data,testing[k],3)
sum=0;
print(result)
for k in range(len(testing_result)):
    if testing_result[k]==result[k]:
        sum+=1
print('accuracy=',sum/len(testing_result)*100)
result.clear()
