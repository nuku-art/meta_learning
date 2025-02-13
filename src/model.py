import torch
import torch.nn as nn
import torch.nn.functional as F

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
  
  def kernel_func(self, query, support, z): # -> scalor

    if support.dim() == 1:
      support = support.unsqueeze(0)
    z_expanded = z.unsqueeze(0).expand(support.shape[0], -1)
    query_expanded = query.unsqueeze(0).expand(support.shape[0], -1)

    query_z = torch.cat((query_expanded, z_expanded), dim=-1)
    support_z = torch.cat((support, z_expanded), dim=-1)

    delta = torch.all(query_z == support_z, dim=-1).to(torch.int)
    k = torch.exp(-torch.norm(self.func_K(query_z)-self.func_K(support_z), p=2, dim=-1)) + self.func_B(z)*delta
    return k
  
  def forward(self, support_set, query_set, device):  # support = batch*(x1,x2,y)   query = batch*(x1,x2)
    Z_list = self.func_Z(support_set)
    Z = torch.mean(Z_list, dim=0)

    if query_set.dim() == 1:
      query_set = query_set.unsqueeze(0)

    additional_z = Z.unsqueeze(0).repeat(support_set.shape[0],1)
    supportX_Z = torch.cat((support_set[:,:2], additional_z), dim=1)
    support_mean = self.func_M(supportX_Z).squeeze()

    y_list = []
    for query in query_set:
      mix_k = torch.tensor([])
      query_XZ = torch.cat((query, Z))
      mean = self.func_M(query_XZ).squeeze()
      support_x = support_set[:,:2]

      mix_k = self.kernel_func(query, support_x, Z)
      
      self_k = torch.zeros(support_x.shape[0],support_x.shape[0])
      for i, x in enumerate(support_x):
        for j, x_prime in enumerate(support_x):
          self_k[i,j] = self.kernel_func(x, x_prime, Z)
      support_y = support_set[:,2]
      self_k = self_k.to(device)
      pred = mean + torch.matmul(mix_k.T, torch.linalg.pinv(self_k)) @  (support_y - support_mean)
      y_list.append(pred)
    y = torch.stack(y_list)
    return y
