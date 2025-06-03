import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

st.title("Mall Customer Segmentation using K-Means")
df=pd.read_csv("Mall_Customers.csv")
st.subheader("📊 Raw data")
st.dataframe(df.head())

x=df[["Annual Income (k$)", "Spending Score (1-100)"]]
scale=StandardScaler()
x_scaled=scale.fit_transform(x)

wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,random_state=40)
    kmeans.fit(x_scaled)
    wcss.append(kmeans.inertia_)

fig,ax=plt.subplots()
ax.plot(range(1,11),wcss,marker='o')
ax.set_title("Elbow Method")
ax.set_xlabel("Number of Clusters")
ax.set_ylabel("WCSS")
st.pyplot(fig)

k=st.slider("Select the number of clusters",2,10,5)

kmeans=KMeans(n_clusters=k,random_state=40)
clusters=kmeans.fit_predict(x_scaled)

df['Cluster']=clusters

fig2,ax2=plt.subplots()
scatter=ax2.scatter(x_scaled[:,0],x_scaled[:,1],c=clusters,cmap="rainbow")
ax2.set_title("Customer Segment")
ax2.set_xlabel("Annual Income (k$)")
ax2.set_ylabel("Spending Score (1-100)")
st.pyplot(fig2)

st.subheader("📋 Clustered Data")
st.dataframe(df[['CustomerID', 'Annual Income (k$)', 'Spending Score (1-100)', 'Cluster']])
