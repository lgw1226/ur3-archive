import os

import pickle
import copy
import warnings
import numpy as np

import gymnasium as gym
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv


class UR3Env(MujocoEnv, utils.EzPickle):
    '''Simulation environment made using gymnasium and mujoco'''

    ##### class variables #####

    # full path to the xml file (MJCF)
    mujoco_xml_full_path = os.path.join(os.path.dirname(__file__), 'assets/ur3/single_ur3_base.xml')
    
    # a frame refers to a step in mojoco simulation
    mujocoenv_frame_skip = 1

    ##### state #####

    # the number of robot and gripper joint coordinates
    ur3_nqpos, gripper_nqpos = 6, 10

    # the number of robot and gripper joint velocities
    ur3_nqvel, gripper_nqvel = 6, 10

    # 4 objects on the table
    objects_nqpos = [7, 7, 7, 7]  # 3 trans, 4 quaternion
    objects_nqvel = [6, 6, 6, 6]  # 3 trans, 3 rotation

    ##### action #####

    # the number of actions for robot and gripper, resp
    ur3_nact, gripper_nact = 6, 2

    # to check collision between objects in simulation, set to True
    ENABLE_COLLISION_CHECKER = False

    ##### initilization #####

    def __init__(self):

        # check collision by copying the simulation and stepping forward
        # collision checker is the part where the copy is generated
        if self.ENABLE_COLLISION_CHECKER:
            self._define_collision_checker_variables()

        self._ezpickle_init()  # pickling related utility library (included in gym)
        self._mujocoenv_init()
        self._check_model_parameter_dimensions()
        self._define_instance_variables()

    ##### hidden methods required for initialization #####

    def _ezpickle_init(self):
        utils.EzPickle.__init__(self)

    def _mujocoenv_init(self):
        '''
        Initialize parent class MujocoEnv with xml path and frame skip
        
        self.model is defined by calling __init__.
        '''

        MujocoEnv.__init__(self, self.mujoco_xml_full_path, self.mujocoenv_frame_skip)

    def _check_model_parameter_dimensions(self):
        '''
        Check if the defined number of variables equals the one from self.model
        
        The number of variables includes the robot, gripper, and objects.
        '''
    
        assert self.ur3_nqpos + self.gripper_nqpos + sum(self.objects_nqpos) == self.model.nq, "# of qpos elements not matching"
        assert self.ur3_nqvel + self.gripper_nqvel + sum(self.objects_nqvel) == self.model.nv, "# of qvel elements not matching"    
        assert self.ur3_nact + self.gripper_nact == self.model.nu, "# of action elements not matching"

    def _define_instance_variables(self):
        '''Define instace variables such as initial posiiton and robot/gripper parameters'''

        # degree to radian
        D2R = np.pi / 180.0

        # initial position of the ur3 robot
        self.init_qpos[0:self.ur3_nqpos] = \
            np.array([-90.0, -90.0, -90.0, -90.0, -135.0, 90.0]) * D2R
        
        # parameters for forward/inverse kinematics of **UR3** robot using Denavit-Hartenberg parameters
        # https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/
        self.kinematics_params = {}

        # there are 2 options choosing 'd' parameter
        # 1. Last frame aligns with (right/left)_ee_link body frame
        # self.kinematics_params['d'] = np.array([0.1519, 0, 0, 0.11235, 0.08535, 0.0819]) # in m
        # 2. Last frame aligns with (right/left)_gripper:hand body frame (only the last element was modified due to the addition of gripper)
        self.kinematics_params['d'] = np.array([0.1519, 0, 0, 0.11235, 0.08535, 0.0819+0.12]) # in m
        self.kinematics_params['a'] = np.array([0, -0.24365, -0.21325, 0, 0, 0]) # in m
        self.kinematics_params['alpha'] =np.array([np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0]) # in rad
        self.kinematics_params['offset'] = np.array([0, 0, 0, 0, 0, 0])  # theta

        # upper and lower bounds for each joint
        self.kinematics_params['ub'] = np.array([2*np.pi for _ in range(6)])
        self.kinematics_params['lb'] = np.array([-2*np.pi for _ in range(6)])

        # transition from world body to base link (right_arm_rotz)
        self.kinematics_params['T_wb_right'] = np.eye(4)
        self.kinematics_params['T_wb_right'][0:3,0:3] = self.sim.data.get_body_xmat('right_arm_rotz').reshape([3,3]).copy()
        self.kinematics_params['T_wb_right'][0:3,3] = self.sim.data.get_body_xpos('right_arm_rotz').copy()

        # pickle the kinematics parameters to the given path if there isn't one already
        path_to_pkl = os.path.join(os.path.dirname(__file__), '../real/ur/dual_ur3_kinematics_params.pkl')
        if not os.path.isfile(path_to_pkl):
            pickle.dump(self.kinematics_params, open(path_to_pkl, 'wb'))

    def _define_collision_checker_variables(self):
        '''Define new variable pointing to the instance itself to check collision'''

        self.collision_env = self

    def _is_collision(self, right_ur3_qpos):
        '''
        Check if there is collision with the given qpos of UR3 robot, return True if there is collision
        
        The method must be run <before> simulating to the next step.
        '''

        is_collision = False

        if self.ENABLE_COLLISION_CHECKER:
            # save original qpos and qvel of the environment to revert to the original state after checking
            qpos_original = self.collision_env.sim.data.qpos.copy()
            qvel_original = self.collision_env.sim.data.qvel.copy()
            
            # qpos and qvel to actually check collision
            qpos = self.collision_env.sim.data.qpos.copy()
            qvel = np.zeros_like(self.collision_env.sim.data.qvel)

            # change qpos of UR3 robot with the given qpos
            qpos[:6] = right_ur3_qpos

            # if there is collision, is_collision = True
            self.collision_env.set_state(qpos, qvel)
            is_collision = self.collision_env.sim.data.nefc > 0

            # set the state to its original qpos and qvel
            self.collision_env.set_state(qpos_original, qvel_original)

        return is_collision

    ##### utility methods #####

    def forward_kinematics_DH(self, q, arm):
        '''
        Compute transition matrix and rotation, translation vectors given joint vector and arm (left/right)
        
        Forward kinematics transforms joint coordinates to cartesian coordinates.
        '''

        assert len(q) == self.ur3_nqpos, "Length of joint coordinates vector q not matching with ur3_nqpos"
        self._define_instance_variables()

        if arm == 'right':
            T_0_i = self.kinematics_params['T_wb_right']
        elif arm == 'left':
            T_0_i = self.kinematics_params['T_wb_left']
        else:
            raise ValueError('Invalid arm type!')
        
        # transition, rotation, translation
        # The matrix and vectors below contains ur3_nqpos + 1 element, each of which from world frame to i-th frame
        T = np.zeros([self.ur3_nqpos+1, 4, 4])
        R = np.zeros([self.ur3_nqpos+1, 3, 3])
        p = np.zeros([self.ur3_nqpos+1, 3])

        # base frame
        T[0,:,:] = T_0_i
        R[0,:,:] = T_0_i[0:3,0:3]
        p[0,:] = T_0_i[0:3,3]

        # from base frame to i-th link body
        for i in range(self.ur3_nqpos):

            # cos, sin offset (theta)
            ct = np.cos(q[i] + self.kinematics_params['offset'][i])
            st = np.sin(q[i] + self.kinematics_params['offset'][i])

            # cos, sin alpha
            ca = np.cos(self.kinematics_params['alpha'][i])
            sa = np.sin(self.kinematics_params['alpha'][i])

            # from i-th link body to i+1-th link body
            T_i_iplus1 = np.array([[ct, -st*ca, st*sa, self.kinematics_params['a'][i]*ct],
                                   [st, ct*ca, -ct*sa, self.kinematics_params['a'][i]*st],
                                   [0, sa, ca, self.kinematics_params['d'][i]],
                                   [0, 0, 0, 1]])
            T_0_i = np.matmul(T_0_i, T_i_iplus1)

            # base frame index is 0 (i = 0)
            T[i+1, :, :] = T_0_i
            R[i+1, :, :] = T_0_i[0:3,0:3]
            p[i+1, :] = T_0_i[0:3,3]

        return R, p, T
    
    def forward_kinematics_ee(self, q, arm):
        '''Compute forward kinematics to the end effector frame'''

        R, p, T = self.forward_kinematics_DH(q, arm)

        return R[-1, :, :], p[-1, :], T[-1, :]
    
    def _jacobian_DH(self, q, arm):
        '''
        Compute jacobian given joint coordinates
        
        The method is required for inverse dynamics.
        '''

        assert len(q) == self.ur3_nqpos, "Length of joint coordinates vector q not matching with ur3_nqpos"

        # small amount of perturbation
        epsilon = 1e-6
        epsilon_inv = 1 / epsilon

        # extract position vector from base frame to ee frame
        _, ps, _ = self.forward_kinematics_DH(q, arm)
        p = ps[-1, :]

        # jacobian computed here only considers position vector
        jac = np.zeros([3, self.ur3_nqpos])

        for i in range(self.ur3_nqpos):

            # compute perturbed values
            q_ = q.copy()
            q_[i] = q_[i] + epsilon  # add perturbation to the given joint coordinates
            _, ps_, _ = self.forward_kinematics_DH(q_, arm)  # forward kinematics (perturbed)
            p_ = ps_[-1, :]  # position (perturbed)
            
            # { p(x+e) - p(x) } / e
            jac[:, i] = (p_ - p) * epsilon_inv

        return jac
    
    def inverse_kinematics_ee(self, ee_pos, null_obj_func, arm,
                              q_init='current', threshold=0.01,
                              threshold_null=0.001, max_iter=100, epsilon=1e-6):
        '''
        Inverse kinematics computes joint coordinates when given end-effector frame cartesian coordinates

        The method computes end-effector frame cartesian coordinates by
        implementing pseudo-inverse of jacobian and null-space approach.
        '''

        # set initial guess
        if arm == 'right':
            if type(q_init).__name__ == 'ndarray': q = q_init.copy()  # use input qpos as initial guess
            elif q_init == 'current': q = self._get_ur3_qpos()[:self.ur3_nqpos]  # use current qpos as initial guess
            elif q_init == 'zero': q = np.zeros([self.ur3_nqpos])  # use zero vector as initial guess
            else: raise ValueError("q_init must be one of the following: ['current', 'zero', numpy.ndarray]")
        elif arm == 'left':  # when using lefthandside UR3 robot (not gonna be used?)
            if type(q_init).__name__ == 'ndarray': q = q_init.copy()
            elif q_init == 'current': q = self._get_ur3_qpos()[self.ur3_nqpos:]
            elif q_init == 'zero': q = np.zeros([self.ur3_nqpos])
            else: raise ValueError("q_init must be one of the following: ['current', 'zero', numpy.ndarray]")
        else:
            raise ValueError('Invalid arm type!')
        
        SO3, x, _ = self.forward_kinematics_ee(q, arm)  # get rotation matrix and position vector from initial guess

        jac = self._jacobian_DH(q, arm)  # compute jacobian from initial guess

        delta_x = ee_pos - x  # position error between desired end position and current position vector
        err = np.linalg.norm(delta_x)  # 2-norm of the position error vector

        # null_obj_func acts as a secondary objective while optimizing for the smallest norm of position error vector
        # can be chosen as whatever the user wants (in this case the function is dependent to the rotational matrix)
        null_obj_val = null_obj_func.evaluate(SO3)

        iter_taken = 0

        while True:
            
            if (err < threshold and null_obj_val < threshold_null) or iter_taken >= max_iter: break
            else: iter_taken += 1

            # pseudo-inverse of jacobian
            jac_dagger = np.linalg.pinv(jac)

            # null space of jacobian
            jac_null = np.eye(self.ur3_nqpos) - np.matmul(jac_dagger, jac)

            phi = np.zeros(self.ur3_nqpos)

            for i in range(self.ur3_nqpos):
                
                # perturb each element of q, leaving others same as original q
                q_perturb = q.copy()
                q_perturb[i] += epsilon

                # get perturbed rotation matrix based on perturbed q (initial guess)
                SO3_perturb, _, _ = self.forward_kinematics_ee(q_perturb, arm)

                # perturbed null objective value
                null_obj_val_perturb = null_obj_func.evaluate(SO3_perturb)

                phi[i] = (null_obj_val_perturb - null_obj_val) / epsilon

            # compute the amount to update
            delta_x = ee_pos - x
            delta_q = np.matmul(jac_dagger, delta_x) - np.matmul(jac_null, phi)
            
            # update joint coordinates vector and clip according to lower and upper bound
            q += delta_q
            q = np.clip(q, self.kinematics_params['lb'], self.kinematics_params['ub'])

            # evaluate null objective function with updated guess
            SO3, x, _ = self.forward_kinematics_ee(q, arm)
            jac = self._jacobian_DH(q, arm)
            null_obj_val = null_obj_func.evaluate(SO3)

            # evaluate position error with updated guess
            err = np.linalg.norm(delta_x)

        if iter_taken == max_iter:
            warnings.warn('Max iteration limit reached! err: %f (threshold: %f), null_obj_err: %f (threshold: %f)'%(err, threshold, null_obj_val, threshold_null), RuntimeWarning)

        return q, iter_taken, err, null_obj_val

    ##### MujocoEnv related utility methods #####