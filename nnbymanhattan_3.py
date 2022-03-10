"""
@author : 201810756 kimdaehwan
@date : 2021.11
@description : Nearest Neighbor Algorithm by Manhattan distance (k=3)
"""
# import pandas, numpy, voronoi
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt
import numpy as np
draft=pd.read_csv('../기말고사 대체과제_1/Nearest/draft.csv')


# manhattan distance
def manhattan(train,test):
    distance=0.0
    for i in range(len(train)):
        distance+=abs(train[i]-test[i])
    return distance
# k-nearest neighbor
def knn(dataset,test,k):
    training = dataset.iloc[:, [1, 2]].values.tolist()
    distance=[0]*len(training)
    for i in range(len(training)):
        distance[i]=manhattan(training[i],test)
    dis=pd.DataFrame(distance,columns=['Distance'])
    data=((pd.concat([dataset,dis],axis=1)).sort_values('Distance',ascending=True)).reset_index(drop=True)
    labels,counts=np.unique(data.loc[0:k-1,['Draft']],return_counts=True)
    maxindex = 0
    max=0
    for i in range(len(counts)):
        if counts[i]>max:
            maxindex=i
            max=counts[i]
    result=labels[maxindex]
    print(test,'result label:',result)
    draft.loc[len(draft)]=[(len(draft)+1),test[0],test[1],result]

#test
print('-------3NN by manhattan--------')
knn(draft,[6.75,3.0],3)
knn(draft,[5.34,6.0],3)
knn(draft,[4.67,8.4],3)
knn(draft,[7.0,7.0],3)
knn(draft,[7.8,5.4],3)


# Voronoi Diagram
points=draft[['Speed','Agility']]
vor=Voronoi(points)
fig=voronoi_plot_2d(vor,point_size=1)
non_draft=draft[draft['Draft']=='No']
draft=draft[draft['Draft']=='Yes']
non_draft_sc=plt.scatter(non_draft['Speed'],non_draft['Agility'],marker='x',color='b',s=50)
draft_sc=plt.scatter(draft['Speed'],draft['Agility'],marker='o',color='r',s=50)
plt.legend((non_draft_sc,draft_sc),('non-draft','draft'),loc='upper left')
plt.title("3nn by Manhattan")
plt.xlabel("Speed")
plt.ylabel("Agility")
plt.show()