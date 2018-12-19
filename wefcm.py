import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import random
import operator
import math
m =1.25
gamma=-100
def initializeMembershipMatrix(n,k):
    U=np.ones((n,k))
    u=np.divide(U,k)
    u=u.tolist()
    return u
def initializeWeight(k,D):
  Weight=list()
  random.seed(100000)
  for i in range(k):
   random_num_list = [random.random() for i in range(D)] 
   summation = sum(random_num_list)
   temp_list = [x/summation for x in random_num_list]
   Weight.append(temp_list)
  return Weight

def initializeWeight1(k,D):
    W=np.ones((k,D))
    w=np.divide(W,D)
    w=w.tolist()
    w[0][1]=0.5
    w[2][5]=0.004
    return w


def calculateClusterCenter(U,sx):
    u=np.array(U)
    um=u**m
    sx=np.array(sx)
    um=um.T
    cntr = um.dot(sx) / np.atleast_2d(um.sum(axis=1)).T
    return cntr    

def updateMembershipValue1(D,membership_mat,k):
   p = float(2/(m-1))
   for i in range(len(membership_mat)):
    d=D[i]
    for j in range(k):	
     den=sum([math.pow(float(d[j]/d[k1]),p) for k1 in range(k)])  
     membership_mat[i][j]=float(1/den)
   return membership_mat

def updateMembershipValue(D,U,k):
   p = float(2/(m-1))
   u=np.power(D,p)
   u1=np.reciprocal(D,dtype=float)
   u2=np.power(u1,p)
   u3=np.sum(u2,axis=1)
   u=u.T
   u4=np.multiply(u,u3)
   u4=u4.T
   u4=np.reciprocal(u4,dtype=float)
   u4=u4.tolist()
   return u4
def calculate_D1(W,C,X,k):
    D=list()
    x=np.array(X)
    w=np.array(W)
    for i in range(k):
     c=C[i];
     c=np.array(c)
     c1=np.subtract(x,c)
     c2=np.square(c1)
     c3=np.multiply(c2,w[i])
     c4=np.sum(c3,axis=1)
     c5=c4.tolist()
     D.append(c5)
    D=np.array(D)
    D=D.T
    D=D.tolist()
    return D     
def calculate_d2(C,U,X,k):
    D2=list()
    x=np.array(X)
    u=np.array(U)
    u=np.power(u,m)
    for i in range(k):
     c1=np.subtract(x,C[i])
     c2=np.square(c1)
     c2=c2.T
     c22=np.array(u[:,i])
     c3=np.multiply(c2,c22)
     c3=c3.T
     c4=np.sum(c3,axis=0)
     D2.append(c4)
    return D2  

'''def updateweight1(W,k,D2,D):
    for i in range(k):
     for j in range(D):
      sum=0.0
      for k1 in range(D):
       sum+=math.exp((-1*D2[i][k1])/gamma)
      W[i][j]=math.exp((-1*D2[i][j])/gamma)/sum
    return W'''
def updateweight(W,D2):
    gamma1=float(-1/gamma)
    D2=np.array(D2)
    D2=np.multiply(D2,gamma1)
    D2=np.exp(D2)
    d3=np.sum(D2,axis=1)
    D2=D2.T
    W=np.divide(D2,d3)
    W=W.T
    W=W.tolist()
    return W    
def WEFCM(Z,k):
 #os.system('clear')
 D=len(Z[0])
 max_iter=160
 print max_iter
 print ('M:= ',m)
 print ('iteration:= ',max_iter)
 ''' print m
 f = open("result1.ods","a+")
 f.write(" %.4f %f %f " %(m, gamma, max_iter))
 f.close()'''
 n=len(Z)
 W=initializeWeight(k,D)
 U=initializeMembershipMatrix(n,k)
 center=calculateClusterCenter(U,Z)
 i=0
 aphselan=np.max(center)
 while(i<=max_iter):
  #print "level6"     
  D1=calculate_D1(W,center,Z,k)
  #print "level7"
  U=updateMembershipValue(D1,U,k)
  #print "level8"
  D2=calculate_d2(center,U,Z,k)
  #print "level9"
  W=updateweight(W,D2)
  #print "level10"
  center1=calculateClusterCenter(U,Z)
  c_d=np.subtract(center1,center)
  c_d=np.square(c_d)
  c_d1=np.sum(c_d,axis=1)
  c_d1=np.sqrt(c_d1)
  aphselan=np.max(c_d1)
  center=center1
  #print "level11"
  #print i
  i+=1
 return U,center
