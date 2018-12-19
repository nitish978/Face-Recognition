import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import pandas as pd
import random
import operator
import math
import timeit
from sklearn import preprocessing
path='orl_faces'
def data_and_labels(imagepaths):
 X=list()
 Y=list()
 for f in imagepaths:	
  img_path=[os.path.join(f,f1) for f1 in os.listdir(f)]
  for i in range(len(img_path)):
   img = cv2.imread(img_path[i])
   #img =cv2.resize(img,(128,128),interpolation=cv2.INTER_CUBIC) #for rnmp dataset 
   gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)	
   X.append(list(gray.flatten()))   
   Y.append(f) 
 return X,Y
def data_collect():
  imagepaths=[os.path.join(path,f) for f in os.listdir(path)]
  X,Y=data_and_labels(imagepaths)
  print len(X[0])
  print len(X[20])
  print len(X[30])
  #X1=preprocessing.scale(X)
  X1=preprocessing.normalize(X)
  X_train,X_test,y_train,y_test=train_test_split(X1,Y, test_size=0.1)
  index=[i for i in range(len(X_train[0]))]
  my_df = pd.DataFrame(X_train)
  my_df.set_index(index)
  size=len(X_train[0])
  my_df1 = pd.DataFrame(y_train)
  my_df[size-1]=my_df1.values
  print my_df[size-1]
  print my_df.columns
  my_df.to_csv('data/90.csv', index=False, header=False)
  my_df11 = pd.DataFrame(X_test)
  my_df11.set_index(index)
  size=len(X_train[0])
  my_df111= pd.DataFrame(y_test)
  my_df11[size-1]=my_df111.values
  my_df11.to_csv('data/10.csv', index=False, header=False)

data_collect()	
