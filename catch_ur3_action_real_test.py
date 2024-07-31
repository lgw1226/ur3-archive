import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

import URBasic

from catch_ur3.envs.catch_ur3 import CatchUR3Env


HOST = '192.168.5.102'  # IP address of the UR3 robot to use
ACC = 0.5
VEL = 0.5


def coordinate_test():

    # setup connection
    robotModel = URBasic.RobotModel()
    robot = URBasic.UrScriptExt(host=HOST, robotModel=robotModel)

    robot.reset_error()

    p = robot.get_actual_tcp_pose()
    
    axis = 1
    increment = -0.2
    delta_p = np.zeros(6)
    delta_p[axis] = increment

    p = delta_p + p
    robot.movej(pose=p, a=ACC, v=VEL)

    robot.close()

def kinematics_test():

    # setup connection
    robotModel = URBasic.RobotModel()
    robot = URBasic.UrScriptExt(host=HOST, robotModel=robotModel)

    robot.reset_error()

    p_target = np.array([0.4, -0.4, 0.75])
    h_target = np.array([0, -1, 1])
    h_target = h_target / np.linalg.norm(h_target)

    # make x-axis of end-effector frame align with ball heading vector

    q_initial = robot.get_actual_joint_positions()

    env = CatchUR3Env()
    q_target, info = env.inverse_kinematics_ee_align_heading(p_target, h_target)

    robot.movej(q=q_target, a=ACC, v=VEL); print('sent')

    env.close()
    robot.close()


def base_test():

    # setup connection
    robotModel = URBasic.RobotModel()
    robot = URBasic.UrScriptExt(host=HOST, robotModel=robotModel)

    robot.reset_error()

    pose = robot.get_actual_tcp_pose()  # x, y, z, rx, ry, rz
    p_base_ee = pose[:3]
    rotation = Rotation.from_rotvec(pose[3:])
    R_base_ee = rotation.as_matrix()
    print(R_base_ee)
    print(p_base_ee)
    
    env = CatchUR3Env()
    env.reset()

    robot.movej(q=env._get_ur3_qpos(), a=ACC, v=VEL)

    R_world_base, p_world_base, _ = env.forward_kinematics_DH(env._get_ur3_qpos())
    R_world_base = R_world_base[0]
    p_world_base = p_world_base[0]
    print(R_world_base)
    print(p_world_base)

    R_world_ee = R_base_ee @ R_world_base
    p_world_ee = p_world_base + R_world_base.T @ p_base_ee

    print(R_world_ee)
    print(p_world_ee)

    R, p, _ = env.forward_kinematics_ee(env._get_ur3_qpos())
    print(R)
    print(p)

    robot.close()
    env.close()




if __name__ == '__main__':

    base_test()