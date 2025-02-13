import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt

class training():
    def __init__(self, device, train_key, test_key, base_dir, YMD, support_num=5, query_num=64):
        self.device = device
        self.train_key = train_key
        self.test_key = test_key
        self.base_dir = base_dir
        self.YMD = YMD
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
            optimizer.zero_grad()
            # chose support set and query set
            support_set, query_set = self.chose_dataset(train_dataset)
            support_set, query_set = support_set.to(self.device), query_set.to(self.device)
            query_input = query_set[:,:2]
            query_label = query_set[:,2]
            # pred = model(support_set, query_input, self.device)
            pred = model(support_set, query_input, self.device)
            loss = criterion(pred, query_label)
            loss.backward()
            optimizer.step()
            # save loss
            loss_recent.append(loss.detach().to('cpu'))
            
            # print statistics
            if i % 10 == 9:    # print every 2000 mini-batches
                print(f'now {i+1} epoch')
                average_loss = sum(loss_recent)/len(loss_recent)
                loss_historty.append(average_loss)
                print(f'[{i + 1}epoch] loss: {average_loss}')
                model_path = f'/workspace/output/model/meta_model_{self.YMD}.pth'
                torch.save(model.state_dict(), model_path)

                # loss tracker
                plt.figure(figsize=(10, 6))
                plt.plot(range(1, len(loss_historty)+1), loss_historty, marker='o', label='Training Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Training Loss Over Epochs')
                plt.legend()
                plt.grid()
                savefig_path = f'{self.base_dir}/Loss_track.png'
                plt.savefig(savefig_path)
                plt.close()