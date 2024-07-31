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

import URBasic

from catch_ur3.envs.catch_ur3 import CatchUR3Env


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
    
    def pull_one(self):

        return self.memory.pop()


def set_ur3(host, q_init):
    '''Establish connection to UR3 robot and move the robot to initial position'''

    robotModel = URBasic.RobotModel()
    robot = URBasic.UrScriptExt(host=host, robotModel=robotModel)

    robot.reset_error()
    robot.movej(q=q_init)

    return robot

    
def ros_callback(data, args):

    buffer = args[0]
    offset = args[1]
    
    wall_time = data.header.stamp.secs + data.header.stamp.nsecs / 1e9
    wall_time = wall_time % 200

    data = torch.tensor([data.pose.position.x - offset[0],
                         data.pose.position.y - offset[1],
                         data.pose.position.z - offset[2],
                         wall_time], dtype=torch.float64)

    buffer.push(data)

def ros_node(ros_topic_name, buffer, offset):

    rospy.Subscriber(
        f'optitrack/{ros_topic_name}/poseStamped',
        PoseStamped,
        ros_callback,
        (buffer, offset)
    )

    rospy.spin()


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

    vx = torch.tensor(savgol_filter(px, len(data), 2, 1, mean_dt))
    vy = torch.tensor(savgol_filter(py, len(data), 2, 1, mean_dt))
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

    x_mask = (0.1 <= px) & (px <= 0.6)
    y_mask = py <= -0.1
    z_mask = (0.4 <= pz) & (pz <= 1.2)
    idx_mask = x_mask & y_mask & z_mask

    vx = savgol_filter(pos_candidates[:,0], num_sample, 1, deriv=1, delta=interval)
    vy = savgol_filter(pos_candidates[:,1], num_sample, 1, deriv=1, delta=interval)
    vz = savgol_filter(pos_candidates[:,2], num_sample, 2, deriv=1, delta=interval)

    filtered_pos_candidates = torch.tensor(np.vstack((px[idx_mask], py[idx_mask], pz[idx_mask])).T)
    filtered_vel_candidates = torch.tensor(np.vstack((vx[idx_mask], vy[idx_mask], vz[idx_mask])).T)

    pos_err_norm = torch.norm(filtered_pos_candidates - ee_pos_now, dim=1)
    idx_min = torch.argmin(pos_err_norm)

    ee_intercept = filtered_pos_candidates[idx_min,:]
    print(f"interception {ee_intercept} | distance {pos_err_norm[idx_min]} | time {t[idx_min].item()}")

    ee_heading = -filtered_vel_candidates[idx_min]
    ee_heading = ee_heading / torch.norm(ee_heading)

    return ee_intercept.numpy(), ee_heading.numpy()


HOST = '192.168.5.102'


def main():

    optitrack_hz = 100
    optitrack_dt = 1 / optitrack_hz

    renew_max_iter = 200
    renew_interval = 10

    num_sample = 9  # must be an odd integer
    buffer = Buffer(num_sample)

    env = CatchUR3Env()
    env.reset()

    q_init = np.array([114.05, -71.41, 106.96, -190.86, -69.83, 19.3]) * np.pi / 180
    robot = set_ur3(HOST, q_init)
    robot_offset = np.array([0, 1.806, 0])

    model = ParabolaModel()

    ros_topic_name = 'glee_ball'
    rospy.init_node('ros_optitrack')
    ros_thread = threading.Thread(target=ros_node, args=(ros_topic_name, buffer, robot_offset), daemon=True)
    ros_thread.start()

    while True:

        data = buffer.pull(num_sample)

        if is_parabolic(data):

            algo_start = time.time()

            px, py, pz, vx, vy, vz, t_offset = get_parabola_ic(buffer.pull(num_sample))
            model.initialize_parameters(px, py, pz, vx, vy, vz)
            model.fit(preprocess_data(buffer.pull(num_sample), t_offset))

            print(f'Parabola time {time.time() - algo_start:.4f}')

            _, ee_pos, _ = env.forward_kinematics_ee(robot.get_actual_joint_positions())  # the robot thinks it's at the origin, but it's actually not
            print(f'FK time {time.time() - algo_start:.4f}')
            ee_pos, ee_heading = find_feasible_ee_pose(ee_pos, model)
            print(f'Feasible time {time.time() - algo_start:.4f}')
            q_target, _ = env.inverse_kinematics_ee_align_heading(ee_pos, ee_heading)  # the result which is joint coordinates is independent to the coordinate system
            print(f'IK time {time.time() - algo_start:.4f}')
            algo_end = time.time()

            print(f'Computation time: {algo_end - algo_start:.4f}')

            robot.movej(q=q_target, a=50, v=50)
            break

    loop_elapsed = 0

    for i in range(renew_max_iter):

        loop_start = time.time()

        if i // renew_interval > 0:

            algo_start = time.time()

            model.fit(preprocess_data(buffer.pull(num_sample), t_offset))

            _, ee_pos, _ = env.forward_kinematics_ee(robot.get_actual_joint_positions())  # the robot thinks it's at the origin, but it's actually not
            ee_pos, ee_heading = find_feasible_ee_pose(ee_pos, model)
            q_target, _ = env.inverse_kinematics_ee_align_heading(ee_pos, ee_heading)  # the result which is joint coordinates is independent to the coordinate system

            algo_end = time.time()

            robot.movej(q=q_target, a=1.4, v=1.05)

            print(algo_end - algo_start)

        loop_end = time.time()
        loop_elapsed = loop_start - loop_end

        if optitrack_dt > loop_elapsed and loop_elapsed > 0:

            time.sleep(optitrack_dt - loop_elapsed)

def robot_test():

    env = CatchUR3Env()
    env.reset()
    q_init = env._get_ur3_qpos()

    robot = set_ur3(HOST, q_init)
    
    # below position and heading are written w.r.t. robot's coordinate system
    ee_pos = np.array([ 0.3109, -0.4053,  0.7415])
    ee_heading = np.array([ 0.0272, -0.4870,  0.8730])
    q_target, _ = env.inverse_kinematics_ee_align_heading(ee_pos, ee_heading)  # the result which is joint coordinates is independent to the coordinate system

    robot.movej(q=q_target)
    # robot.servoj(q=q_target)

if __name__ == '__main__':

    main()