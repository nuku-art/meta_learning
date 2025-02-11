import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.datasets import fetch_california_housing
from sklearn.cluster import KMeans
import random
import matplotlib.pyplot as plt

class meta_learning(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_Z_1 = nn.Linear(3,256)
        self.fc_Z_2 = nn.Linear(256, 256)
        self.fc_B_1 = nn.Linear(256, 256)
        self.fc_B_2 = nn.Linear(256, 1)
        self.fc_M_1 = nn.Linear(258, 256)
        self.fc_M_2 = nn.Linear(256, 1)
        self.fc_K_1 = nn.Linear(258, 256)
        self.fc_K_2 = nn.Linear(256, 256)
        self.dropout = nn.Dropout(0.1)

    def func_Z(self, o):    # o = [x,y] in R^3 (subset)
        o = F.relu(self.fc_Z_1(o))
        o = self.dropout(o)
        o = self.fc_Z_2(o)
        return o
  
    def func_B(self, p):    # p = z in R^64
        p = F.relu(self.fc_B_1(p))
        p = self.dropout(p)
        p = self.fc_B_2(p)
        return p
  
    def func_M(self, q):    # q = [x,z] in R^66
        q = F.relu(self.fc_M_1(q))
        q = self.dropout(q)
        q = self.fc_M_2(q)
        return q

    def func_K(self, r):    # r = [x,z] in R^66
        r = F.relu(self.fc_K_1(r))
        r = self.dropout(r)
        r = self.fc_K_2(r)
        return r
  
    def kernel_func(self, x1, x2, z): # -> scalor
        x1_z = torch.cat((x1, z))
        x2_z = torch.cat((x2, z))
        delta = int(torch.all(x1_z == x2_z))
        k = torch.exp(-torch.norm(self.func_K(x1_z)-self.func_K(x2_z), p=2)) + self.func_B(z)*delta
        return k
  
    def forward(self, support_set, query_set, device):  # support = batch*(x1,x2,y)   query = batch*(x1,x2)
        # calcurate task representation Z
        Z_list = []
        for i in range(support_set.shape[0]):
            z = self.func_Z(support_set[i])
            Z_list.append(z)
        Z = torch.mean(torch.stack(Z_list), dim=0)
        
        # calcurate support set mean m
        support_mean = torch.tensor([]).to(device)
        for support_vector in support_set:
            supX_Z = torch.cat((support_vector[:2],Z))
            sup_mean = self.func_M(supX_Z)
            support_mean = torch.cat((support_mean, sup_mean))

        if query_set.dim() == 1:
            query_set = query_set.unsqueeze(0)
        y_list = []
        for query in query_set:
            mix_k = torch.tensor([]).to(device)
            query_XZ = torch.cat((query, Z))
            # calcurate mean of GP f_m([x,z])
            mean = self.func_M(query_XZ)
            # calcurate kernel of GP k
            for support_vector in support_set:
                kernel = self.kernel_func(query, support_vector[:2], Z)
                mix_k = torch.cat((mix_k, kernel))
            # calcurete support set covariance K
            support_x = support_set[:,:2]
            self_k = torch.zeros(support_x.shape[0],support_x.shape[0]).to(device)
            for i, x in enumerate(support_x):
                for j, x_prime in enumerate(support_x):
                    self_k[i,j] = self.kernel_func(x, x_prime, Z)
            support_y = support_set[:,2]
            # prediction 
            pred = mean + torch.matmul(mix_k.T, torch.linalg.inv(self_k)) @  (support_y - support_mean)
            y_list.append(pred)
        y = torch.stack(y_list)
        y = torch.squeeze(y)
        return y
    
class training():
    def __init__(self, device, train_key, test_key, support_num=5, query_num=64):
        self.device = device
        self.train_key = train_key
        self.test_key = test_key
        self.support_num = support_num
        self.query_num = query_num
    
    def chose_dataset(self, dataset):
        # select region
        use_key = random.choice(self.train_key)
        choiced_region_data = dataset[dataset['Cluster']==use_key]
        # select annotation
        annotation_data = choiced_region_data.drop(columns=['Latitude','Longitude','Cluster'])
        annotation_columns_list = annotation_data.columns.tolist()
        choiced_annotation = random.choice(annotation_columns_list)
        this_epoch_data = choiced_region_data[['Latitude','Longitude',choiced_annotation]]
        numpy_data = this_epoch_data.to_numpy()
        torch_data = torch.from_numpy(numpy_data).to(torch.float32)
        total_data_num = self.support_num + self.query_num
        rand_index = torch.randint(0,torch_data.size(0),(total_data_num,))
        support_set = torch_data[rand_index[:self.support_num]]
        query__set = torch_data[rand_index[self.support_num:]]
        return support_set, query__set
    
    def optim(self, model, train_dataset, epoch):
        if next(model.parameters()).device != self.device:
            model.to(self.device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        loss_recent = []
        loss_historty = []
        for i in range(epoch):
            print(f'now {i} epoch')

            optimizer.zero_grad()
            # chose support set and query set
            support_set, query_set = self.chose_dataset(train_dataset)
            support_set, query_set = support_set.to(self.device), query_set.to(self.device)
            query_input = query_set[:,:2]
            query_label = query_set[:,2]
            pred = model(support_set, query_input, self.device)
            loss = criterion(pred, query_label)
            loss.backward()
            optimizer.step()
            # save loss
            loss_recent.append(loss.detach().to('cpu'))
            
            # print statistics
            if i % 10 == 9:    # print every 2000 mini-batches
                average_loss = sum(loss_recent)/len(loss_recent)
                loss_historty.append(average_loss)
                print(f'[{i + 1}epoch] loss: {average_loss}')
                model_path = '/workspace/output/model/meta_model.pth'
                torch.save(model.state_dict(), model_path)

                # loss tracker
                plt.figure(figsize=(10, 6))
                plt.plot(range(1, len(loss_historty)+1), loss_historty, marker='o', label='Training Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Training Loss Over Epochs')
                plt.legend()
                plt.grid()
                savefig_path = '/workspace/output/Loss_track.png'
                plt.savefig(savefig_path)
                plt.close()


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


