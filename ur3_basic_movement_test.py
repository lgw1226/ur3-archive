import URBasic

import numpy as np

from catch_ur3.envs.catch_ur3 import CatchUR3Env


HOST = '192.168.5.102'  # IP address of the UR3 robot to use
ACC = 0.5
VEL = 0.5


def ur3_connection_test():
    '''Test connection status and check basic commands.'''

    robotModel = URBasic.RobotModel()
    robot = URBasic.UrScriptExt(host=HOST, robotModel=robotModel)

    ur3_reset_state = robot.reset_error()

    if ur3_reset_state: print('Connection established')
    else: print('Connection failed')

    qpos = robot.get_actual_joint_positions()
    tcp_pos = robot.get_actual_tcp_pose()

    print(f"Joint coordinate: {qpos}")
    print(f"Tool Center Point (TCP) pose: {tcp_pos}")

    robot.close()

def ExampleExtendedFunctions():
    '''
    This is an example of an extension to the Universal Robot script library. 
    How to update the force parameters remote via the RTDE interface, 
    hence without sending new programs to the controller.
    This enables to update force "realtime" (125Hz)  
    '''
    robotModel = URBasic.RobotModel()
    robot = URBasic.UrScriptExt(host=host, robotModel=robotModel)

    print('forcs_remote')
    robot.set_force_remote(task_frame=[0., 0., 0.,  0., 0., 0.], selection_vector=[0,0,1,0,0,0], wrench=[0., 0., 20.,  0., 0., 0.], f_type=2, limits=[2, 2, 1.5, 1, 1, 1])
    robot.reset_error()
    a = 0
    upFlag = True
    while a<3:
        pose = robot.get_actual_tcp_pose()
        if pose[2]>0.1 and upFlag:
            print('Move Down')
            robot.set_force_remote(task_frame=[0., 0., 0.,  0., 0., 0.], selection_vector=[0,0,1,0,0,0], wrench=[0., 0., -20.,  0., 0., 0.], f_type=2, limits=[2, 2, 1.5, 1, 1, 1])
            a +=1
            upFlag = False
        if pose[2]<0.0 and not upFlag:
            print('Move Up')
            robot.set_force_remote(task_frame=[0., 0., 0.,  0., 0., 0.], selection_vector=[0,0,1,0,0,0], wrench=[0., 0., 20.,  0., 0., 0.], f_type=2, limits=[2, 2, 1.5, 1, 1, 1])
            upFlag = True    
    robot.end_force_mode()
    robot.reset_error()
    robot.close()
        

if __name__ == '__main__':

    env = CatchUR3Env()
    env.reset()
    q_init = env._get_ur3_qpos()
    p_init = env._get_ur3_tcp()

    robotModel = URBasic.RobotModel()
    robot = URBasic.UrScriptExt(host=HOST, robotModel=robotModel)

    robot.reset_error()
    robot.movej(q_init)

    q_target = np.array([114.05, -71.41, 106.96, -190.86, -69.83, 19.3]) * np.pi / 180
    p_target = np.array([ 0.15961214, -0.25763657,  0.82949956])
    
    # robot.movej(q_target)
    # robot.movej(q_init)

    robot.movel(pose=p_target)
    robot.movel(pose=p_init)
    