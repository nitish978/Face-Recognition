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
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
path='orl_faces'
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def data_and_labels(imagepaths):
 class_labels=list()
 data =list()
 for f in imagepaths:	
  img_path=[os.path.join(f,f1) for f1 in os.listdir(f)]
  for i in img_path:
   img = cv2.imread(i,0)
   pca = PCA(n_components=2)
   principalComponents = pca.fit_transform(img)
   principalDf = pd.DataFrame(data = principalComponents)
   a = np.array(principalDf)
   data.append(list(a.flatten()))   
   class_labels.append(f)
 return data,class_labels

def normalized_data(data):
 data1=np.transpose(data) 
 for i in range(len(data[0])):
  normalized_X=preprocessing.normalize([data1[i]])
  data1[i]=normalized_X
 data=np.transpose(data1) 
 return data

def sirf_feature_extraction(data):
  pca = PCA(n_components=len(data))
  principalComponents = pca.fit_transform(np.transpose(data))
  principalDf = pd.DataFrame(data = principalComponents)
  a = np.array(principalDf)
  return a

def data_collect():
  imagepaths=[os.path.join(path,f) for f in os.listdir(path)]
  data,labels=data_and_labels(imagepaths)
  data1=normalized_data(data)
  #data2=sirf_feature_extraction(data1)
  return data1,labels

  


