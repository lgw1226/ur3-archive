import threading
import time
from collections import deque

import numpy as np

from scipy.signal import savgol_filter

import torch
import torch.nn as nn
import torch.optim as optim

import rospy
from geometry_msgs.msg import PoseStamped

from catch_ur3.envs.catch_ur3 import CatchUR3Env

import cv2


class ParabolaModel(nn.Module):

    def __init__(self):
        super(ParabolaModel, self).__init__()

        self.g = -9.81

        self.fc_x = nn.Linear(1, 1, dtype=torch.float64, bias=True)
        self.fc_y = nn.Linear(1, 1, dtype=torch.float64)
        self.fc_z = nn.Linear(1, 1, dtype=torch.float64)

        self.optimizer = optim.SGD(self.parameters(), lr=0.05)
        self.criterion = nn.L1Loss()

    def initialize_parameters(self, px, py, pz, vx, vy, vz):

        self.fc_x.weight = torch.nn.Parameter(torch.tensor([[vx]], dtype=torch.float64, requires_grad=True))  # v_x
        self.fc_x.bias = torch.nn.Parameter(torch.tensor([[px]], dtype=torch.float64, requires_grad=True))  # p_x

        self.fc_y.weight = torch.nn.Parameter(torch.tensor([[vy]], dtype=torch.float64, requires_grad=True))  # v_y
        self.fc_y.bias = torch.nn.Parameter(torch.tensor([[py]], dtype=torch.float64, requires_grad=True))  # p_y

        self.fc_z.weight = torch.nn.Parameter(torch.tensor([[vz]], dtype=torch.float64, requires_grad=True))  # v_z
        self.fc_z.bias = torch.nn.Parameter(torch.tensor([[pz]], dtype=torch.float64, requires_grad=True))  # p_z

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

            if torch.mean(loss) < eps: break

        return loss, idx_iter + 1
    
class Buffer():

    def __init__(self, maxlen=20):

        self.memory = deque(maxlen=maxlen)

    def __len__(self):

        return len(self.memory)

    def push(self, data):

        self.memory.append(data)

    def pull(self, num_sample, timeout=10):

        timeout_count = 0

        while len(self.memory) < num_sample:

            time.sleep(0.001)

            timeout_count +=1
            if timeout_count >= (timeout / 0.001):
                raise TimeoutError

        return torch.stack([self.memory.popleft() for _ in range(num_sample)])


def is_parabolic(data):

    target_acc = torch.tensor([0, 0, -9.8])
    norm_err_threshold = 1

    t = data[:,3]
    dt = t[1:] - t[:-1]
    mean_dt = torch.mean(dt)

    px = data[:,0]
    py = data[:,1]
    pz = data[:,2]

    ax = torch.tensor(savgol_filter(px, len(data), 2, 2, mean_dt))
    ay = torch.tensor(savgol_filter(py, len(data), 2, 2, mean_dt))
    az = torch.tensor(savgol_filter(pz, len(data), 2, 2, mean_dt))

    mean_acc = torch.mean(torch.stack([ax, ay, az]), dim=1)
    norm_err = torch.norm(mean_acc - target_acc)

    if norm_err <= norm_err_threshold:
        return True
    else:
        return False

def get_parabola_ic(data):

    t = data[:,3]
    dt = t[1:] - t[:-1]
    mean_dt = torch.mean(dt)

    px = data[:,0]
    py = data[:,1]
    pz = data[:,2]

    vx = torch.tensor(savgol_filter(px, len(data), 1, 1, mean_dt))
    vy = torch.tensor(savgol_filter(py, len(data), 1, 1, mean_dt))
    vz = torch.tensor(savgol_filter(pz, len(data), 2, 1, mean_dt))

    init_px = torch.mean(px)
    init_py = torch.mean(py)
    init_pz = torch.mean(pz)

    init_vx = torch.mean(vx)
    init_vy = torch.mean(vy)
    init_vz = torch.mean(vz)

    init_t = torch.mean(t)

    return init_px, init_py, init_pz, init_vx, init_vy, init_vz, init_t

def preprocess_data(data, t):
    '''Pass data through savgol filter and offset time'''

    dt = 0.02

    data[:,0] = torch.tensor(savgol_filter(data[:,0], len(data), 1, delta=dt))
    data[:,1] = torch.tensor(savgol_filter(data[:,1], len(data), 1, delta=dt))
    data[:,2] = torch.tensor(savgol_filter(data[:,2], len(data), 2, delta=dt))
    data[:,3] = data[:,3] - t

    return data

def find_feasible_ee_pose(ee_pos_now, model):

    num_sample = 9  # to estimate value via sg filter

    start = 0
    end = 1
    interval = 0.001
    t = torch.arange(start, end, interval, dtype=torch.float64).unsqueeze(1)
    
    with torch.no_grad():
        pos_candidates = model(t)

    px = savgol_filter(pos_candidates[:,0], num_sample, 1, delta=interval)
    py = savgol_filter(pos_candidates[:,1], num_sample, 1, delta=interval)
    pz = savgol_filter(pos_candidates[:,2], num_sample, 2, delta=interval)

    x_mask = (-0.1 <= px) & (px <= 0.6)
    y_mask = py <= -0.1
    z_mask = (0.3 <= pz) & (pz <= 1)
    idx_mask = x_mask & y_mask & z_mask

    vx = savgol_filter(pos_candidates[:,0], num_sample, 1, deriv=1, delta=interval)
    vy = savgol_filter(pos_candidates[:,1], num_sample, 1, deriv=1, delta=interval)
    vz = savgol_filter(pos_candidates[:,2], num_sample, 2, deriv=1, delta=interval)

    filtered_pos_candidates = torch.tensor(np.vstack((px[idx_mask], py[idx_mask], pz[idx_mask])).T)
    filtered_vel_candidates = torch.tensor(np.vstack((vx[idx_mask], vy[idx_mask], vz[idx_mask])).T)

    pos_err_norm = torch.norm(filtered_pos_candidates - ee_pos_now, dim=1)
    idx_min = torch.argmin(pos_err_norm)

    ee_intercept = filtered_pos_candidates[idx_min,:]
    # print(f"interception {ee_intercept} | distance {pos_err_norm[idx_min]} | time {t[idx_min].item()}")

    ee_heading = -filtered_vel_candidates[idx_min]
    ee_heading = ee_heading / torch.norm(ee_heading)
    # print(ee_heading)

    return ee_intercept.numpy(), ee_heading.numpy()


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

env = CatchUR3Env(render_mode='rgb_array')

num_episode = 1

frames = []

for idx_episode in range(num_episode):

    print(f'starting a new episode {idx_episode + 1}')

    buffer = Buffer(10)
    num_sample = 5  # must be an odd integer

    is_parabola = False
    is_target = False

    obs, info = env.reset()
    init_q = info['ur3_qpos']

    for idx_step in range(150):

        if len(buffer) >= num_sample and not is_parabola:

            data = buffer.pull(num_sample)
            is_parabola = is_parabolic(data)

            if is_parabola:

                print('parabolic trajectory found')
                traj_step = idx_step

                px, py, pz, vx, vy, vz, t_offset = get_parabola_ic(data)

                model = ParabolaModel()
                model.initialize_parameters(px, py, pz, vx, vy, vz)
                loss, _ = model.fit(preprocess_data(data, t_offset))

                _, ee_pos, _ = env.forward_kinematics_ee(info['ur3_qpos'])
                ee_pos, ee_heading = find_feasible_ee_pose(ee_pos, model)
                q_target, _ = env.inverse_kinematics_ee_align_heading(ee_pos, ee_heading)

                print(f'parabolic trajectory loss {loss:.4f}')

        if is_parabola:
            
            if (traj_step - idx_step) % 10 == 0 and traj_step != idx_step:  # renew trajectory every 5 steps

                data = buffer.pull(num_sample)
                loss, _ = model.fit(preprocess_data(data, t_offset))

                _, ee_pos, _ = env.forward_kinematics_ee(info['ur3_qpos'])
                ee_pos, ee_heading = find_feasible_ee_pose(ee_pos, model)
                q_target, _ = env.inverse_kinematics_ee_align_heading(ee_pos, ee_heading)

            if info['ball_pos'][2] <= 0.3:
                action = env.servoj(info['ur3_qpos'], 0, 0)
            else:
                action = env.servoj(q_target, 0, 0)

        else:

            action = env.servoj(init_q, 0, 0)
        

        obs, reward, terminated, truncated, info = env.step(action)
        frame = env.render()
        frames.append(frame)

        ball_pos = torch.tensor(np.append(info['ball_pos'], info['time']), dtype=torch.float64)
        buffer.push(ball_pos)

env.close()

frames = np.array(frames)

size = 480, 480
fps = 50
duration = len(frames) / fps

out = cv2.VideoWriter('mujoco_simulation.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), True)
for i in range(int(fps * duration)):
    frame = frames[i]
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    out.write(frame)
out.release()