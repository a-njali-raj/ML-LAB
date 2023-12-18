from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

lb=load_breast_cancer()
x=lb.data
y=lb.target

kmeans=KMeans(n_clusters=3,random_state=42)
kmeans.fit(x)

cluster_labels=kmeans.labels_
print(cluster_labels)
centroids=kmeans.cluster_centers_
print(centroids)

plt.scatter(x[:, 0],x[:, 1],c=cluster_labels,cmap='viridis',marker='o',edgecolor='black')
plt.scatter(centroids[:, 0],centroids[:, 1],marker='*',s=200,c='red',label='Centroids')
plt.xlabel(lb.feature_names[0])
plt.ylabel(lb.feature_names[1])
plt.title('KMeans Clustering of breast_cancer dataset')
plt.legend()
plt.show()