#!/usr/bin/env python
# coding: utf-8

# In[110]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.metrics.cluster import v_measure_score
import seaborn as sns


# In[243]:


df = pd.read_csv("C:/Users/devin/Downloads/test_image.txt", header=None)


# In[244]:


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
        probs = cluster_dist_df[['dist']]/max(cluster_dist_df['dist'])
        centroids.append(df.sample(1,weights=probs.squeeze()))
    return centroids
'''
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
'''
def assign_centroid_smart(k_centroids, point):
    min = distance.cosine(k_centroids[0],point)
    k=1
    for i in range(0,len(k_centroids)):
        dist = distance.cosine(k_centroids[i],point)
        if(dist < min):
            min=dist
            k=i + 1
    return [k,min]    

def assign_centroid(k_centroids, point):
    min = distance.cosine(k_centroids[0],point)
    k=1
    for i in range(0,len(k_centroids)):
        dist = distance.cosine(k_centroids[i],point)
        if(dist < min):
            min=dist
            k=i + 1
    return k

def new_centroids(cluster):
    means = cluster.median()
    new_centroid=cluster.iloc[0]
    mins =  distance.cosine(new_centroid,means)
    for i in range(0,len(cluster)):
        dist = distance.cosine(cluster.iloc[i],means)
        if(dist <= mins):
            mins=dist
            new_centroid = cluster.iloc[i]
    return new_centroid


def kmeans(df,k):
    clusters = []
    centroids = initial_centroids(df,k)
    new_centroid = smart_initial_centroids(df,k)
    while(np.array_equal(np.array(centroids),np.array(new_centroid))):
        centroids=initial_centroids(df,k)
        
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
            df['cluster'] = clusters
        else:
            error = 0
            for i in range(0,len(df)):
                error = error + distance.cosine(df.iloc[i],centroids[df.iloc[i,784] -1])**2
            stop=False
    return error
            
            


# In[245]:


print(kmeans(df,10) )


# In[246]:


with open('image_out.txt', 'w') as f:
    for i in range(0,len(df)):
        f.write(str(df.iloc[i,784]) + "\n")


# In[166]:


sse= []
for i in range(2,21):
    print(i)
    sse.append(kmeans(df,i))


# In[247]:


plt.plot(range(2,21),sse)
plt.ylabel('SSE')
plt.xlabel('K')
plt.grid()

