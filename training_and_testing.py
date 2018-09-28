import numpy as np
from orl_data_extraction import data_collect
from wefcm import WEFCM
from sklearn.model_selection import train_test_split
from sklearn.metrics.cluster import adjusted_rand_score as ari
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
from sklearn.neighbors import KNeighborsClassifier as knn
import timeit

def getpredicted_labels(U,n,Y):
 predicted_labels=list()
 for i in range(n):
  predicted_labels.append(Y[U[i].index(max(U[i]))])
 return predicted_labels

def total_labels(class_labels):
 total_sample=list() 
 for i in class_labels:
  if i not in total_sample:
    total_sample.append(i)
 return total_sample   
  
def label_cluster(X,class_labels,k,center,total_sample):
 neigh=knn(n_neighbors=1)
 neigh.fit(X,class_labels)
 Y=neigh.predict(center)
 return Y

def test_data(X_test,center,y1):
 neigh=knn(n_neighbors=1)
 neigh.fit(center,y1)
 Y=neigh.predict(X_test)
 return Y

def training_and_testing():
 k=40	
 X,labels=data_collect()
 X_train,X_test,y_train,y_test=train_test_split(X,labels,test_size=0.2)
 start=timeit.default_timer()
 U,C=WEFCM(X_train,k)
 pre_labels=getpredicted_labels(U,len(X_train),y_train)
 ts=total_labels(y_train)
 y1_train=label_cluster(X_train,y_train,k,C,ts)
 pre_labels=getpredicted_labels(U,len(y_train),y1_train)
 stop=timeit.default_timer()
 print len(pre_labels)
 print len(y_train)
 print nmi(y_train,pre_labels)
 print ari(y_train,pre_labels)
 print ('run time:= ',stop-start)
 y1_test=test_data(X_test,C,y1_train)
 accuracy=(float(np.sum(y1_test==y_test)))/len(y_test)
 print ('accuracy:= ',accuracy)
 print nmi(y_test,y1_test)

training_and_testing() 
