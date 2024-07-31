import time
import threading
from collections import deque

import numpy as np
import torch
from scipy.signal import savgol_filter

import rospy
from geometry_msgs.msg import PoseStamped

    
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


robot_offset = np.array([0, 2.079, 0])

num_sample = 9  # must be an odd integer
buffer = Buffer(num_sample)

ros_topic_name = 'glee_cube'
rospy.init_node('ros_optitrack')
ros_thread = threading.Thread(target=ros_node, args=(ros_topic_name, buffer, robot_offset), daemon=True)
ros_thread.start()

while True:

    time.sleep(0.2)

    data = buffer.pull(num_sample)

    if is_parabolic(data):

        print('parabola found!')