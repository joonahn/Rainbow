import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class TransitionEvaluator:
    def __init__(self, args, agent=None):
        self.device = args.device
        self.batch_size = args.batch_size
        self.pointnet = PointNetCls()
        self.pointnet.cuda()
        self.optimizer = optim.Adam(self.pointnet.parameters(), lr=0.001, betas=(0.9, 0.999))
        self.history = args.history_length
        self.discount = args.discount
        self.n = args.multi_step
        self.cnt = 0
        self.priority_activation = args.priority_activation
        self.n_step_scaling = torch.tensor([self.discount ** i for i in range(self.n)], dtype=torch.float32)  # Discount-scaling vector for n-step returns
        self.cnn = agent.online_net.convs
        self.enable_cnn = args.cnn

    def extract_feats(self, states, actions, rewards, next_states):
        if self.enable_cnn:
            states = states.view(-1, self.history, 84, 84)
            states = self.cnn(states).detach().view(-1, 32, 3136)
            states = states.permute(0, 2, 1) # (set_cnt, 3136, 32)

            next_states = next_states.view(-1, self.history, 84, 84)
            next_states = self.cnn(next_states).detach().view(-1, 32, 3136)
            next_states = next_states.permute(0, 2, 1) # (set_cnt, 3136, 32)
            data = torch.cat([states, next_states, actions, rewards], 1)
            data = data.view(-1, 6274, 32)

            # free memory
            del states, next_states
            torch.cuda.empty_cache()
        else:
            states = states[:, -1]
            states = states.view(-1, 32, 7056).permute(0, 2, 1)
            data = torch.cat([states, actions, rewards], 1)
            data = data.view(-1, 7058, 32)
        return data

    def learn(self, epi_transitions, delta_reward, mem):
        if len(epi_transitions) < self.batch_size:
            print("cannot learn because of short epi_transition length!", len(epi_transitions))
            return None
        self.pointnet.train()
        self.optimizer.zero_grad()
        s, a, r, next_s = mem.get_sample_by_indices(np.array(epi_transitions))
        data = self.extract_feats(s, a, r, next_s)
        predict = self.pointnet(data)
        loss = F.mse_loss(predict, torch.tensor(delta_reward, dtype=torch.float32, device=self.device).repeat(*predict.shape))
        loss.backward()
        self.optimizer.step()
        del s, a, r, next_s, data
        torch.cuda.empty_cache()

    def update_mem_priority(self, epi_transitions, mem):
        if len(epi_transitions) < self.batch_size:
            print("cannot learn because of short epi_transition length!", len(epi_transitions))
            return None
        self.pointnet.eval()
        s, a, r, next_s = mem.get_sample_by_indices(np.array(epi_transitions))
        data = self.extract_feats(s, a, r, next_s)
        predict = self.pointnet(data).detach().cpu().numpy()
        for i in range(data.size()[0]):
            if self.priority_activation == 'sigmoid':
                mem.update_value_by_indices(epi_transitions[i*self.batch_size:(i+1) * self.batch_size], 1 / (1 + np.exp(-predict[i])))
            elif self.priority_activation == 'exponential':
                mem.update_value_by_indices(epi_transitions[i*self.batch_size:(i+1) * self.batch_size], np.exp(predict[i]))
            elif self.priority_activation == 'relu':
                mem.update_value_by_indices(epi_transitions[i*self.batch_size:(i+1) * self.batch_size], np.maximum(np.zeros(predict[i].shape),predict[i]))
            else:
                raise Exception("wrong priority activation: " + self.priority_activation)
        del s, a, r, next_s, data
        torch.cuda.empty_cache()

class PointNetfeat(nn.Module):
    def __init__(self, enable_cnn=None):
        super(PointNetfeat, self).__init__()
        if enable_cnn:
            self.conv1 = torch.nn.Conv1d(7058, 128, 1)
        else:
            self.conv1 = torch.nn.Conv1d(6274, 128, 1)
        self.conv2 = torch.nn.Conv1d(128, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.enable_cnn = enable_cnn

    def forward(self, x):

        x = F.relu(self.bn1(self.conv1(x)))

        trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        return x

class PointNetCls(nn.Module):
    def __init__(self, k=1, feature_transform=False, enable_cnn=None):
        super(PointNetCls, self).__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(enable_cnn)
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