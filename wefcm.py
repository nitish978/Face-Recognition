import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import random
import operator
import math
m = 1.7
gamma=-32
def initializeMembershipMatrix(n,k):
    membership_mat = list()
    for i in range(n):
     random_num_list = [random.random() for i in range(k)] 
     summation = sum(random_num_list)
     temp_list = [x/summation for x in random_num_list]
     membership_mat.append(temp_list)
    return membership_mat

def initializeWeight(k,D):
  Weight=list()
  for i in range(k):
   random_num_list = [random.random() for i in range(D)] 
   summation = sum(random_num_list)
   temp_list = [x/summation for x in random_num_list]
   Weight.append(temp_list)
  return Weight
def calculateClusterCenter(membership_mat,X,k):
    cluster_mem_val = zip(*membership_mat)
    cluster_centers = list()
    for j in range(k):
      u=list(cluster_mem_val[j])
      xraised = [e ** m for e in u]
      denominator = sum(xraised)
      temp_num = list()
      x=list()
      for i in range(len(X)):
      	 x=X[i]
      	# print x
         prod = [xraised[i] * val for val in x]
         temp_num.append(prod)
      numerator = map(sum,zip(*temp_num))

      center = [z/denominator for z in numerator]
      cluster_centers.append(center)
    return cluster_centers

def calculate_D1(Weight,cluster_center,X,k):
    D=list()
    for i in range(len(X)):
     x=X[i]
     distance=[map(operator.sub,x,cluster_center[j]) for j in range(k)]
     dis=[map(lambda x: x**2, distance[j]) for j in range(k)]
     dis1=[map(operator.mul,Weight[j],dis[j]) for j in range(k)]
     dis2=[sum(dis1[j]) for j in range(k)]
     D.append(dis2)  
    return D
def updateMembershipValue(D,membership_mat,k):
   p = float(2/(m-1))
   for i in range(len(membership_mat)):
    d=D[i]
    for j in range(k):	
     den=sum([math.pow(float(d[j]/d[k1]),p) for k1 in range(k)])  
     membership_mat[i][j]=float(1/den)
   return membership_mat

def calculate_d2(Weight,center,U,X,k,D):
    D2=list() 
    u=zip(*U)
    for i in range(k):
     u_pow_alpha=[math.pow(x,m) for x in u[i]]
     dis=[map(operator.sub,X[i],center[i]) for j in range(len(X))]
     dis1=[map(lambda x: x**2, dis[j]) for j in range(len(X))]
     dis11=zip(*dis1)
     dis2=[map(lambda x,y: x*y,u_pow_alpha,dis11[j]) for j in range(D)]
     dis3=[sum(dis2[x]) for x in range(D)]
     D2.append(dis3)
    return D2  

def updateweight(W,k,D2,D):
    for i in range(k):
     for j in range(D):
      sum=0.0
      for k1 in range(D):
       sum+=math.exp((-1*D2[i][k1])/gamma)
      W[i][j]=math.exp((-1*D2[i][j])/gamma)/sum
    return W
def WEFCM(Z,k):
 D=len(Z[0])
 max_iter=100
 n=len(Z)
 my_df = pd.DataFrame(Z)
 my_df.to_csv('out.csv', index=False, header=False)
 W=initializeWeight(k,D)
 U=initializeMembershipMatrix(n,k)
 center=calculateClusterCenter(U,Z,k)
 i=0
 while(i<100):
  #print "level6"     
  D1=calculate_D1(W,center,Z,k)
  #print "level7"
  U=updateMembershipValue(D1,U,k)
  #print "level8"
  D2=calculate_d2(W,center,U,Z,k,D)
  #print "level9"
  W=updateweight(W,k,D2,D)
  #print "level10"
  center=calculateClusterCenter(U,Z,k)
  #print "level11"
  print i
  i+=1
 return U,center