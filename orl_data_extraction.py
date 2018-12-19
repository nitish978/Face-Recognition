import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import random
import operator
import math
import timeit
from sklearn import preprocessing
from sklearn.decomposition import PCA,KernelPCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.manifold import LocallyLinearEmbedding as lle
from sklearn.kernel_approximation import RBFSampler
df_full =pd.read_csv("data/90.csv")
df_full=df_full.sample(frac=1)
columns =list(df_full.columns)
features = columns[0:len(columns)-1]
class_labels=list(df_full[columns[-1]])
df = df_full[features]


df_full1 =pd.read_csv("data/10.csv")
df_full1=df_full1.sample(frac=1)
columns1 =list(df_full1.columns)
features1 = columns1[0:len(columns1)-1]
class_labels1=list(df_full1[columns1[-1]])
df1 = df_full1[features1]
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def data_and_labels(imagepaths):
 X=list()
 Y=list()
 for f in imagepaths:	
  img_path=[os.path.join(f,f1) for f1 in os.listdir(f)]
  for i in range(len(img_path)):
   img = cv2.imread(img_path[i])
   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)		
   X.append(list(gray.flatten()))   
   Y.append(f) 
 return X,Y
def feature_reduction_using_rbf(X): 
  rbf_feature=RBFSampler(gamma=0.8,n_components=20)
  X=rbf_feature.fit_transform(X)
  return X
def feature_reduction_using_lle(X,X1):
     clf=lle(n_components=50)
     clf.fit(X)
     X_transformed=clf.transform(X)
     X_transformed1=clf.transform(X1)
     return X_transformed,X_transformed1
def feature_extraction_using_decision_tree(X,Y,X1):#x=x_train,Y=Y_train,X1=X_test
     clf=ExtraTreesClassifier(n_estimators=3)
     clf.fit(X,Y)
     model=SelectFromModel(clf,prefit=True)
     X_new=model.transform(X)
     X_new1=model.transform(X1)
     return X_new,X_new1
def feature_extraction_using_pca(X_train,X_test):
  pca =PCA(n_components=25)
  pca.fit(X_train)
  principalComponents = pca.transform(X_train)
  principalDf = pd.DataFrame(data = principalComponents)
  a = np.array(principalDf)
  principalComponents1= pca.transform(X_test)
  principalDf1 = pd.DataFrame(data = principalComponents1)
  a1 = np.array(principalDf1)
  return a,a1

def data_collect_for_wefcm():
  imagepaths=[os.path.join(path,f) for f in os.listdir(path)]
  X,Y=data_and_labels(imagepaths)
  #X=np.array(df)
  X1=preprocessing.scale(X)
  X1=preprocessing.normalize(X1)
  #X_train,X_test,y_train,y_test=train_test_split(X1,Y, test_size=0.2)
  X1=feature_reduction_using_rbf(X1)	
  return X1,class_labels

def data_collect_for_arwefcm():
  imagepaths=[os.path.join(path,f) for f in os.listdir(path)]
  X,Y=data_and_labels(imagepaths)
  X1=preprocessing.scale(X)
  return X1,Y
def data_collection_from_file():
	X=np.array(df)
	X=X.tolist()
	X1=preprocessing.scale(X)
	X1=preprocessing.normalize(X1) 
    
	X2=np.array(df1)
	X2=X2.tolist()
	X3=preprocessing.scale(X2)
	X3=preprocessing.normalize(X3)
	X1,X2=feature_reduction_using_lle(X1,X3)
	return X1,X2,class_labels,class_labels1
def data_collection_from_test_file():
	X=np.array(df1)
	X=X.tolist()
	X1=preprocessing.scale(X)
	X1=preprocessing.normalize(X1)
	#X1=feature_reduction_using_lle(X1)
	return X1,class_labels1	




  

