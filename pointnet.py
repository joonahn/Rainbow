import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class TransitionEvaluator:
    def __init__(self, args):
        self.device = args.device
        self.batch_size = args.batch_size
        self.pointnet = PointNetCls()
        self.pointnet.cuda()
        self.optimizer = optim.Adam(self.pointnet.parameters(), lr=0.001, betas=(0.9, 0.999))
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)
        self.history = args.history_length
        self.discount = args.discount
        self.n = args.multi_step
        self.n_step_scaling = torch.tensor([self.discount ** i for i in range(self.n)], dtype=torch.float32)  # Discount-scaling vector for n-step returns

    def learn(self, epi_transitions, delta_reward, mem):
        if len(epi_transitions) < self.batch_size:
            print("cannot learn because of short epi_transition length!", len(epi_transitions))
            return None
        else:
            print("else, length:", len(epi_transitions))
        states, actions, rewards = mem.get_sample_by_indices(np.array(epi_transitions))
        self.pointnet.train()
        self.optimizer.zero_grad()
        data = torch.cat([states, actions, rewards], 1)
        data = data.view(-1, 7058, 32)
        predict = self.pointnet(data)
        loss = F.mse_loss(predict, torch.tensor(delta_reward, dtype=torch.float32, device=self.device).repeat(*predict.shape))
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_mem_priority(self, epi_transitions, mem):
        if len(epi_transitions) < self.batch_size:
            print("cannot learn because of short epi_transition length!", len(epi_transitions))
            return None
        states, actions, rewards = mem.get_sample_by_indices(np.array(epi_transitions))
        data = torch.cat([states, actions, rewards], 1)
        data = data.view(-1, 7058, 32)
        predict = self.pointnet(data)
        for i in range(data.size()[0]):
            mem.update_value_by_indices(epi_transitions[i*self.batch_size:(i+1) * self.batch_size], nn.Sigmoid()(predict[i]))


class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False):
        super(PointNetfeat, self).__init__()
        self.conv1 = torch.nn.Conv1d(7058, 128, 1)
        self.conv2 = torch.nn.Conv1d(128, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform

    def forward(self, x):
        n_pts = x.size()[2]
        x = F.relu(self.bn1(self.conv1(x)))

        trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        return x

class PointNetCls(nn.Module):
    def __init__(self, k=1, feature_transform=False):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return x