import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import fetch_california_housing
from sklearn.cluster import KMeans
import random
import scipy
from src.model import meta_learning
import datetime
import os

YMD = str(datetime.date.today()).replace('-','')
base_dir = f'/workspace/output/{YMD}'
os.makedirs(base_dir, exist_ok=True)

log_file_path = f'{base_dir}/inference_log.txt'
with open(log_file_path, mode='w') as f:
  f.write('only inference')

# load dataset
california_housing_data = fetch_california_housing()
exp_data = pd.DataFrame(california_housing_data.data, columns=california_housing_data.feature_names)
tar_data = pd.DataFrame(california_housing_data.target, columns=['HousingPrices'])
data = pd.concat([exp_data, tar_data], axis=1)

## clustering
use_feature = data[['Latitude', 'Longitude']]
kmeans = KMeans(n_clusters=50, random_state=87)
clusters = kmeans.fit_predict(use_feature)
data['Cluster'] = clusters
unique, counts = np.unique(clusters, return_counts=True)
cluster_counts = dict(zip(unique, counts))
with open(log_file_path, mode='w') as f:
  f.write('\n')
  f.write(f'cluster count: {cluster_counts}\n')

## visualize
sns.scatterplot(x='Latitude', y='Longitude', hue='Cluster', data=data, palette='Set1', legend=False)
plt.title('K-means Clustering Results')
plt.savefig(f'{base_dir}/cluster.png')

## normalize annotation
# columns_norm = data.columns.difference(['Latitude','Longitude','Cluster'])
columns_norm = data.columns.difference(['Cluster'])
data[columns_norm] = (data[columns_norm] - data[columns_norm].mean()) / data[columns_norm].std()
with open(log_file_path, mode='w') as f:
  f.write("normalize latitude and longitude\n")

## prepair train_dataset and test_dataset
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

with open(log_file_path, mode='a') as f:
    f.write("\n")
    f.write(f'number of cluster whos member is lower than thresholdcount: {count}\n')

train_data = data[data['Cluster'].isin(train_key)]
train_data = train_data.drop(columns=['HousingPrices'])
test_data = data[data['Cluster'].isin(test_key)]
test_data = test_data[['Latitude', 'Longitude', 'HousingPrices', 'Cluster']]

## load model
model = meta_learning()
path = '/workspace/output/model/meta_model_20250213.pth'
model.load_state_dict(torch.load(path, weights_only=True))
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

# inference
model.to(device).eval()

epoch = 500
label_set = []
pred_set = []

for i in range(epoch):
  if i % 10 == 9:
    print(f'now {i+1} epoch')

  ## select region
  use_key = random.choice(test_key)
  choiced_region_data = test_data[test_data['Cluster']==use_key]
  now_epoch_data = choiced_region_data.drop(columns=['Cluster'])

  ## select support*5 and query*15
  numpy_data = now_epoch_data.to_numpy()
  torch_data = torch.from_numpy(numpy_data).to(torch.float32)
  usedata_num = 20
  # rand_index = torch.randint(0,torch_data.size(0),(6,))
  rand_index = torch.randperm(torch_data.size(0))
  support = torch_data[rand_index[:5]].to(device)
  query = torch_data[rand_index[5:usedata_num]].to(device)
  label = query[:,2]

  ## prediction
  with torch.no_grad():
    pred = model(support, query[:,:2], device)

  ## save result
  label_set.append(label.detach().to('cpu'))
  pred_set.append(pred.detach().to('cpu'))

label_tensor = torch.stack(label_set)
pred_tensor = torch.stack(pred_set)

## calcurate MSEloss
mseloss = torch.norm(label_tensor - pred_tensor, p=2)**2 / (label_tensor.shape[0] * label_tensor.shape[1])
with open(log_file_path, mode='w') as f:
  f.write('\n')
  f.write(f'MSE Loss is {mseloss}')

label_vector = torch.flatten(label_tensor)
pred_vector = torch.flatten(pred_tensor)

## visualize result
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(pred_vector, label_vector, color="blue", alpha=0.6, label="Data")
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_aspect('equal')
ax.plot([-3, 3], [-3, 3], color="black", linestyle="--", linewidth=2, label="y = x")
ax.set_title('Accuracy', fontsize=14)
ax.set_xlabel('Predicted Value', fontsize=12)
ax.set_ylabel('True Value', fontsize=12)
plt.savefig(f'{base_dir}/accuracy.png')