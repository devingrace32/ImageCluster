#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance
import seaborn as sns


# In[317]:


iris = pd.read_csv("C:/Users/devin/Downloads/iris.txt", names=['sl','sw','pl','pw'], sep=' ')


# In[397]:


def initial_centroids(df,k):
    centroids=[]
    for i in range(0,k):
        centroids.append(df.sample(1))
    return centroids

def smart_initial_centroids(df,k):
    centroids=[]
    cluster_dist_df=pd.DataFrame(columns=['cluster','dist'])
    centroid = df.sample()
    centroids.append(centroid)
    for j in range(1,k):
        for i in range(0,len(df)):
            cluster_dist_df.loc[i] = assign_centroid_smart(centroids,df.iloc[i])
        max_dist_indx = cluster_dist_df[['dist']].idxmax()
        centroids.append(df.iloc[max_dist_indx])
    return centroids

def assign_centroid_smart(k_centroids, point):
    min = distance.euclidean(k_centroids[0],point)
    k=1
    for i in range(0,len(k_centroids)):
        dist = distance.euclidean(k_centroids[i],point)
        if(dist < min):
            min=dist
            k=i + 1
    return [k,min]    

def assign_centroid(k_centroids, point):
    min = distance.euclidean(k_centroids[0],point)
    k=1
    for i in range(0,len(k_centroids)):
        dist = distance.euclidean(k_centroids[i],point)
        if(dist < min):
            min=dist
            k=i + 1
    return k

def new_centroids(cluster):
    means = cluster.mean()
    new_centroid=cluster.iloc[0]
    min =  distance.euclidean(new_centroid,means)
    for i in range(0,len(cluster)):
        dist = distance.euclidean(cluster.iloc[i],means)
        if(dist < min):
            min=dist
            new_centroid = cluster.iloc[i]
    return new_centroid

def convergence(vec1,vec2):
    return np.array_equal(vec1,vec2)

def stopping_crit(previous_centroids,centroids):
    stop=False
    count=0
    for i in range(0,len(centroids)):
        if(convergence(np.array(previous_centroids[i]),np.array(centroids[i]))):
            count+=1
    if(count == len(centroids)):
        return True
    return False
    
def kmeans(df,k):
    clusters = []
    centroids = initial_centroids(df,k)
    new_centroid = smart_initial_centroids(df,k)
    while(np.array_equal(np.array(centroids),np.array(new_centroid))):
        centroids=initial_centroids(df,k)
        print('x')
        
     
    for i in range(0,len(df)):
        clusters.append(assign_centroid(centroids,df.loc[i])) 
    df['cluster'] = clusters   
    stop=True
    while(stop):
        new_centroid = []
        for i in range(0,k):
            new_centroid.append(new_centroids(df[df.cluster==i+1]))
        if(not np.array_equal(np.array(centroids),np.array(new_centroid))):
            centroids=new_centroid
            for i in range(0,len(df)):
                clusters[i]=assign_centroid(centroids,df.iloc[i])
            #print(clusters)
            df['cluster'] = clusters
        else:
            stop=False
    return df
            
        


# In[398]:


df = kmeans(iris,3) 


# In[406]:


with open('iris_output.txt', 'w') as f:
    for i in range(0,len(df)):
        f.write(str(df.iloc[i,4]) + "\n")

