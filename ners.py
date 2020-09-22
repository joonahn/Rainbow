import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class TransitionEvaluator:
    def __init__(self, args, agent):
        self.device = args.device
        self.agent = agent
        self.discount = args.discount
        self.batch_size = args.batch_size
        self.ners_update_sample_size = 128
        self.ners = NERS(input_dim=14116)
        self.ners.cuda()
        self.optimizer = optim.Adam(self.ners.parameters(), lr=0.0001, betas=(0.9, 0.999))
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)
        self.history = args.history_length
        self.discount = args.discount
        self.n = args.multi_step
        self.priority_activation = args.priority_activation
        self.n_step_scaling = torch.tensor([self.discount ** i for i in range(self.n)], dtype=torch.float32)  # Discount-scaling vector for n-step returns

    def learn(self, epi_transitions, delta_reward, mem):
        if len(epi_transitions) < self.ners_update_sample_size:
            print("cannot learn because of short epi_transition length!", len(epi_transitions))
            return None
        self.ners.train()
        selected_transitions = np.random.choice(epi_transitions, self.ners_update_sample_size)
        states, actions, rewards, next_states = mem.get_sample_by_indices(np.array(selected_transitions))
        with torch.no_grad():
            td_error = rewards + self.discount * self.agent.target_net(next_states).sum(2).max(1)\
                - self.agent.online_net(states).sum(2)[:, actions]
            q_val = rewards + self.discount * self.agent.target_net(states).sum(2).max(1)
        states = states[:, -1]
        next_states = next_states[: -1]
        self.optimizer.zero_grad()
        data = torch.cat([states, actions, rewards, next_states, td_error, q_val], 1)
        data = data.view(-1, 14116)
        # REINFORCE algorithm
        _, _, predict = self.ners(data)
        loss = (-predict.log() * delta_reward).sum()
        loss.backward()
        self.optimizer.step()
        del states, actions, rewards, next_states, td_error, q_val, data
        torch.cuda.empty_cache()

    def update_mem_priority(self, epi_transitions, mem):
        if len(epi_transitions) < self.batch_size:
            print("cannot learn because of short epi_transition length!", len(epi_transitions))
            return None
        self.ners.eval()
        states, actions, rewards, next_states = mem.get_sample_by_indices(np.array(epi_transitions))
        with torch.no_grad():
            # rewards: (32, 1)
            # target_net: [32] -> max는 이상한 객체를 return 하므로 [0]으로 값만 추출해야함
            # online_net: [32, 1] -> how to select action??
            # print("rewards:", rewards.shape) 
            # print("target_net:", self.agent.target_net(next_states).sum(2).max(1, keepdim=True)[0].shape)
            # print("online_net:", torch.gather(self.agent.online_net(states).sum(2), 1, actions.long()))
            td_error = rewards + self.discount * self.agent.target_net(next_states).sum(2).max(1, keepdim=True)[0]\
                - torch.gather(self.agent.online_net(states).sum(2), 1, actions.long())
            q_val = rewards + self.discount * self.agent.target_net(states).sum(2).max(1, keepdim=True)[0]
        states = states[:, -1].view(-1, 7056)
        next_states = next_states[:, -1].view(-1, 7056)
        self.optimizer.zero_grad()
        data = torch.cat([states, actions, rewards, next_states, td_error, q_val], 1)
        data = data.view(-1, 14116)
        # REINFORCE algorithm
        with torch.no_grad():
            priorities, _, _ = self.ners(data)
            priorities = priorities.cpu().numpy()[:, 0]
            print(priorities.shape)
        mem.update_value_by_indices(epi_transitions, priorities)
        del states, actions, rewards, next_states, td_error, q_val, data
        torch.cuda.empty_cache()

class NERS(nn.Module):
    def __init__(self, input_dim):
        super(NERS, self).__init__()
        # hyperparams
        self.alpha = 0.5
        self.beta = 0.4

        # input_dim -> 256, 512, 256, 128
        self.local_net = nn.Sequential(
          nn.Linear(input_dim, 256),
          nn.ReLU(),
          nn.Linear(256, 512),
          nn.ReLU(),
          nn.Linear(512, 256),
          nn.ReLU(),
          nn.Linear(256, 128),
          nn.ReLU(),
        )
        self.global_net = nn.Sequential(
          nn.Linear(input_dim, 256),
          nn.ReLU(),
          nn.Linear(256, 512),
          nn.ReLU(),
          nn.Linear(512, 256),
          nn.ReLU(),
          nn.Linear(256, 128),
          nn.ReLU(),
        )

        # 256, 128, 64 -> 1
        self.score_net = nn.Sequential(
          nn.Linear(256, 128),
          nn.ReLU(),
          nn.Linear(128, 64),
          nn.ReLU(),
          nn.Linear(64, 1),
          nn.ReLU(),
        )

    def forward(self, x):
        # input: D(I)
        local_feat = self.local_net(x)
        global_feat = torch.mean(self.global_net(x), 0).repeat(local_feat.shape[0], 1)
        concat_feat = torch.cat((local_feat, global_feat), 1)
        logits = self.score_net(concat_feat)
        print(logits)
        logits = logits ** self.alpha
        priorities = logits / (torch.sum(logits) + 1e-4)
        weights = (1/(priorities * priorities.shape[0])) ** self.beta
        return priorities, weights, logits
        
