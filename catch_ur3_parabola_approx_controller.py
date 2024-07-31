from copy import copy
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from catch_ur3.envs.catch_ur3 import CatchUR3Env


class DataBuffer:
    
    def __init__(self, data_size: list, maxlen=5):

        self.maxlen = maxlen
        self.buffer = np.zeros([self.maxlen] + data_size)

        self.idx = 0
        self.full = False

    def push(self, data):

        self.buffer[self.idx] = data

        self.idx = (self.idx + 1) % self.maxlen
        self.full = self.full or self.idx == 0

    def get_data(self):
        return torch.tensor(self.buffer)
    

class ParabolaModel(nn.Module):

    def __init__(self):
        super(ParabolaModel, self).__init__()

        self.g = -9.81

        self.fc_x = nn.Linear(1, 1, dtype=torch.float64, bias=True)
        self.fc_y = nn.Linear(1, 1, dtype=torch.float64)
        self.fc_z = nn.Linear(1, 1, dtype=torch.float64)

        self.initialize_parameters()

        self.optimizer = optim.SGD(self.parameters(), lr=0.05)
        self.criterion = nn.L1Loss()

    def initialize_parameters(self):

        self.fc_x.weight = torch.nn.Parameter(torch.tensor([[0]], dtype=torch.float64, requires_grad=True))  # v_x
        self.fc_x.bias = torch.nn.Parameter(torch.tensor([[0]], dtype=torch.float64, requires_grad=True))  # p_x

        self.fc_y.weight = torch.nn.Parameter(torch.tensor([[2.5]], dtype=torch.float64, requires_grad=True))  # v_y
        self.fc_y.bias = torch.nn.Parameter(torch.tensor([[-3]], dtype=torch.float64, requires_grad=True))  # p_y

        self.fc_z.weight = torch.nn.Parameter(torch.tensor([[2]], dtype=torch.float64, requires_grad=True))  # v_z
        self.fc_z.bias = torch.nn.Parameter(torch.tensor([[1.5]], dtype=torch.float64, requires_grad=True))  # p_z

    def forward(self, t):

        pred_x = self.fc_x(t)
        pred_y = self.fc_y(t)
        pred_z = self.fc_z(t) + 0.5 * self.g * t ** 2

        return torch.cat((pred_x, pred_y, pred_z), dim=1)
    
    def fit(self, data, eps=0.01, max_iter=200):

        p = data[:,0:3]
        t = data[:,3].unsqueeze(1)

        for idx_iter in range(max_iter):

            pred_p = self.forward(t)
            loss = self.criterion(pred_p, p)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if torch.mean(loss) < eps:
                break
        
        print(idx_iter + 1)

        return loss


def tidy_center_print(val='', terminal_width=80):

    print(f"{val:-^{terminal_width}}")

def tidy_key_val_print(key, val, terminal_width=80):
    
    leftover_width = terminal_width - len(key)

    print(f"{key}{val:>{leftover_width}.4f}")


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

env = CatchUR3Env()

num_episode = 1
len_ball_pos_sequence = 10  # number of coordinates that make up a sequence

tidy_center_print('Info')
tidy_key_val_print("Time elapsed between two consecutive simulation frames (sec)", env.dt)
tidy_key_val_print("Number of coordinates used in computing parabolic trajectory (#)", len_ball_pos_sequence)
tidy_center_print()

for idx_episode in range(num_episode):

    # create new model for each episode
    model = ParabolaModel()

    obs, info = env.reset()

    ball_pos_sequence = DataBuffer([4], maxlen=len_ball_pos_sequence)
    ball_pos_sequence.push(np.append(info['ball_pos'], info['time']))

    for idx_step in range(50):
        
        action = np.zeros(6)
        obs, reward, terminated, truncated, info = env.step(action)

        ball_pos_sequence.push(np.append(info['ball_pos'], info['time']))

        if idx_step + 1 >= len_ball_pos_sequence:

            loss = model.fit(ball_pos_sequence.get_data())
            print(loss.item())
 
env.close()

