import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import random

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model2.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)


class QTrainer:
    def __init__(self, model1, model2, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model1 = model1
        self.model2 = model2
        self.optimizer = optim.Adam(model1.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done, )

        # 1: predicted Q values with current state
        pred1 = self.model1(state)
        pred2 = self.model2(state)

        target1 = pred1.clone()
        target2 = pred2.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                if random.random() > 0.5:
                    Q_new = reward[idx] + self.gamma * torch.max(self.model1(next_state[idx]))
                    
                else:
                	Q_new = reward[idx] + self.gamma * torch.max(self.model2(next_state[idx]))
            target1[idx][torch.argmax(action[idx]).item()] = Q_new 
            target2[idx][torch.argmax(action[idx]).item()] = Q_new 

            
    
        # 2: Q_new = r + y * max(next_predicted Q value) -> only do this if not done
        # pred.clone()
        # preds[argmax(action)] = Q_new
        self.optimizer.zero_grad()
        loss = self.criterion((target1+target2)/2, (pred1+pred2)/2)
        loss.backward()

        self.optimizer.step()



