import numpy as np
import pandas as pd
from orl_data_extraction import data_collect_for_arwefcm
from orl_data_extraction import data_collect_for_wefcm
from orl_data_extraction import data_collection_from_file
from orl_data_extraction import data_collection_from_test_file
from ARWFCM import ARWEFCM
from wefcm import WEFCM
from fcm import FCM
from ofcm import oFCM
from ofcm import initializeMembershipMatrix
from ofcm import updatemembershipvalue
from wefcm import calculateClusterCenter
from sklearn.model_selection import train_test_split
from sklearn.metrics.cluster import adjusted_rand_score as ari
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.neighbors import RadiusNeighborsClassifier as knn1
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import timeit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from nltk.classify import accuracy as ac
nbr=3
def getpredicted_labels(U,n):
 predicted_labels=list()
 for i in range(n):
  predicted_labels.append(U[i].index(max(U[i])))
 return predicted_labels   
  
def label_cluster(X,class_labels,center):
 clf=knn(n_neighbors=nbr,algorithm='kd_tree')
 #clf=tree.DecisionTreeRegressor()
 #clf =SVC(kernel='rbf', class_weight='balanced')
 clf.fit(X,class_labels)
 Y=clf.predict(center)
 return Y

def test_data(X_test,center,y1): 	
 clf=knn(n_neighbors=1,algorithm='auto')
 #neigh.fit(center,y1)
 #Y=neigh.predict(X_test)
 #clf=RandomForestClassifier()
 #clf =SVC(kernel='rbf', class_weight='balanced')
 #clf=tree.DecisionTreeRegressor()
 clf.fit(center,y1)
 Y=clf.predict(X_test)
 return Y

def training_and_testing():
 k=40
 #X_train,y_train=data_collect_for_wefcm()
 X_train,X_test,y_train,y_test=data_collection_from_file()
 #print np.shape(X)
 #X_train,X_test,y_train,y_test=train_test_split(X,Y, test_size=0.2)
 start=timeit.default_timer()
 #U=ARWEFCM(X_train,k)
 print (np.shape(X_train))
 U,C=WEFCM(X_train,k)
 #X_test,y_test=data_collection_from_test_file()
 #this is ofcm 
 #print "c start"
 #C=calculateClusterCenter(U,X_train,k)
 #my_df = pd.DataFrame(C)
 #my_df.to_csv('out.csv',index=False, header=False)
 #print C[0:2]
 #print "c end"
 y1_train=label_cluster(X_train,y_train,C)
 pre_labels=getpredicted_labels(U,len(y_train))
 stop=timeit.default_timer()
 print ('run time:= ',stop-start)
 r1 = nmi(y_train,pre_labels)
 r2 = ari(y_train,pre_labels)
 print ('NMI:= ',r1)
 print ('ARI:= ',r2)
 #print y1_train 
 #print len(X_train)
 y1_test=test_data(X_test,C,y1_train) 
 #print C 
 accuracy=(float(np.sum(y1_test==y_test)))/len(y_test)
 print ('accuracy:= ',accuracy)
 #print(classification_report(y_test, y1_test, target_names=y_test))	
 '''f = open("result1.ods","a+")
 f.write("%f   %.10f %.10f %.10f \n" %(nbr,accuracy, r1, r2))
 f.close()'''

training_and_testing() 
