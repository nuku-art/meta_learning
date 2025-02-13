import numpy as np
import pandas as pd
import torch
from sklearn.datasets import fetch_california_housing
from sklearn.cluster import KMeans
import src.model as meta_learning
import src.training as training

# load california housing data
california_housing_data = fetch_california_housing()
exp_data = pd.DataFrame(california_housing_data.data, columns=california_housing_data.feature_names)
tar_data = pd.DataFrame(california_housing_data.target, columns=['HousingPrices'])
data = pd.concat([exp_data, tar_data], axis=1)

# clustering data to decide region
use_feature = data[['Latitude', 'Longitude']]
kmeans = KMeans(n_clusters=50, random_state=87)
clusters = kmeans.fit_predict(use_feature)
data['Cluster'] = clusters
unique, counts = np.unique(clusters, return_counts=True)
cluster_counts = dict(zip(unique, counts))
# arrange data for meta-learning
threshold = 100
count = 0
test_key = []
train_key = []
for key, value in zip(cluster_counts.keys(), cluster_counts.values()):
  if value < threshold:
    count += 1
    test_key.append(key)
  else:
    train_key.append(key)
print(f'number of train region: {50 - count}')
print(f'number of test region: {count}')
train_data = data[data['Cluster'].isin(train_key)]
train_data = train_data.drop(columns=['HousingPrices'])
# normalize train_data
columns_norm = train_data.columns.difference(['Latitude','Longitude','Cluster'])
train_data[columns_norm] = (train_data[columns_norm] - train_data[columns_norm].mean()) / train_data[columns_norm].std()
print(train_data.head())

test_data = data[data['Cluster'].isin(test_key)]
test_data = test_data[['Latitude', 'Longitude', 'HousingPrices', 'Cluster']]
print(test_data.head())

# training
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
epoch = 5000
model = meta_learning()
train = training(device=device, train_key=train_key, test_key=test_key)
train.optim(model=model, train_dataset=train_data, epoch=epoch)

# inference
model.eval()


