import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report

dataset = datasets.load_iris()
data = pd.DataFrame({
    'x': dataset.data[:, 0],
    'y': dataset.data[:, 1],
    'cluster' : dataset.target
})

centroids = {}
for i in range(3):
    result_list = []
    result_list.append(data.loc[data['cluster'] == i]['x'].mean())
    result_list.append(data.loc[data['cluster'] == i]['y'].mean())

    centroids[i] = result_list

fig = plt.figure(figsize=(5, 5))
plt.scatter(data['x'], data['y'], c=dataset.target)
plt.xlabel('Spea1 Length', fontsize=18)
plt.ylabel('Sepal Width', fontsize=18)

colmap = {0: 'r', 1: 'g', 2: 'b'}
for i in range(3):
    plt.scatter(centroids[i][0],centroids[i][1], color=colmap[i])
plt.show()

fig = plt.figure(figsize=(5, 5))
plt.scatter(data['x'], data['y'], c=dataset.target, alpha = 0.3)
colmap = {0: 'r', 1: 'g', 2: 'b'}
col = [0,1]
for i in centroids.keys():
    plt.scatter(centroids[i][0],centroids[i][1], c=colmap[i], edgecolor='k')
plt.show()

def assignment(df, centroids):
    for i in range(3):
        # sqrt((x1 - x2)^2 + (y1 - y2)^2)
        df['distance_from_{}'.format(i)] = (
            np.sqrt(
                (df['x'] - centroids[i][0]) ** 2
                + (df['y'] - centroids[i][1]) ** 2
            )
        )
    centroid_distance_cols = ['distance_from_{}'.format(i) for i in centroids.keys()]
    df['closest'] = df.loc[:, centroid_distance_cols].idxmin(axis=1)
    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_')))
    df['color'] = df['closest'].map(lambda x: colmap[x])
    return df

data = assignment(data, centroids)

fig = plt.figure(figsize=(5, 5))
plt.scatter(data['x'], data['y'], color=data['color'], alpha=0.3)
for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i], edgecolor='k')

plt.show()

def update(k):
    for i in range(3):
        centroids[i][0] = np.mean(data[data['closest'] == i]['x'])
        centroids[i][1] = np.mean(data[data['closest'] == i]['y'])
    return k

centroids = update(centroids)

fig = plt.figure(figsize=(5, 5))

for i in centroids.keys():
    plt.scatter(*centroids[i], color=colmap[i], edgecolor='k')

plt.show()

data = assignment(data, centroids)

fig = plt.figure(figsize=(5, 5))
plt.scatter(data['x'], data['y'], color=data['color'], alpha=0.3)
for i in centroids.keys():
    plt.scatter(centroids[i][0],centroids[i][1], color=colmap[i], edgecolor='k')
plt.show()

while True:
    closest_centroids = data['closest'].copy(deep=True)
    centroids = update(centroids)
    data = assignment(data, centroids)
    if closest_centroids.equals(data['closest']):
        break

fig = plt.figure(figsize=(5, 5))
plt.scatter(data['x'], data['y'], color=data['color'])
for i in centroids.keys():
    plt.scatter(centroids[i][0],centroids[i][1], color=colmap[i], edgecolor='k')

plt.show()


url = "iris.csv"
colnames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']
i = pd.read_csv(url, names=colnames)

i['Class'] = pd.Categorical(i["Class"])
i["Class"] = i["Class"].cat.codes

X = i.values[:, 0:4]
y = i.values[:, 4]



# Number of clusters
kmeans = KMeans(n_clusters=3)
# Fitting the input data
kmeans = kmeans.fit(X)
# Getting the cluster labels
labels = kmeans.predict(X)
# Centroid values
centroids = kmeans.cluster_centers_



target_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

print(classification_report(i['Class'],kmeans.labels_,target_names=target_names))

