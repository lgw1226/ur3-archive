import os
import pickle
from copy import copy

import numpy as np
from scipy.spatial.transform import Rotation

import mujoco
import gymnasium as gym
from gymnasium.utils import EzPickle
from gymnasium.envs.mujoco import MujocoEnv

class CatchUR3Env(MujocoEnv, EzPickle):

    metadata = {
        'render_modes': ['human', 'rgb_array', 'depth_array'],
        'render_fps': 50
    }

    ##### MujocoEnv class variables #####

    # full path to the xml file (MJCF)
    mujoco_xml_full_path = os.path.join(os.path.dirname(__file__), 'assets/mjcf/ur3_catch.xml')

    # a frame refers to a step in mujoco simulation
    mujoco_env_frame_skip = 10

    ##### state configuration #####
    
    # the number of robot coordinates and velocities
    ur3_nqpos = 6
    ur3_nqvel = 6

    # the number of ball coordinates and velocities
    ball_nqpos = 7
    ball_nqvel = 6

    # set observation space low and high bound
    dim_obs = ur3_nqpos + ball_nqpos + ur3_nqvel + ball_nqvel

    ur3_nqpos_bound = np.pi
    ur3_nqvel_bound = 20
    ball_nqpos_bound = 20
    ball_nqvel_bound = 20

    high_bound = np.concatenate([
        np.ones(ur3_nqpos) * ur3_nqpos_bound,
        np.ones(ball_nqpos) * ball_nqpos_bound,
        np.ones(ur3_nqvel) * ur3_nqvel_bound,
        np.ones(ball_nqvel) * ball_nqvel_bound
    ])

    low_bound = - high_bound
    
    observation_space = gym.spaces.Box(low_bound, high_bound, (dim_obs,), dtype=np.float64)

    ##### action configuration #####
    
    # ur3 actions
    ur3_nact = 6

    # enable collision checking if True
    ENABLE_COLLISION_CHECKER = False

    # limit torque of each joint
    ur3_torque_limit = np.array([50.0, 50.0, 25.0, 10.0, 10.0, 10.0])

    ##### initialization #####

    def __init__(self, render_mode=None, camera_id=0):

        # rendering options
        self.render_mode = render_mode
        self.camera_id = camera_id
        
        # check collision by copying the simulation and stepping forward
        # collision checker is the part where the copy is generated
        if self.ENABLE_COLLISION_CHECKER:
            self._define_collision_checker_variables()

        self._ezpickle_init()  # pickling related utility library (included in gym)
        self._mujocoenv_init()
        self._check_model_parameter_dimensions()
        self._define_kinematic_parameters()

        # required for computation of PID control input
        self.ur3_err_integ = 0

    def _ezpickle_init(self):

        EzPickle.__init__(self)

    def _mujocoenv_init(self):
        '''
        Initialize parent class MujocoEnv with xml path and frame skip
        
        self.model is defined by calling __init__.
        '''

        MujocoEnv.__init__(
            self,
            self.mujoco_xml_full_path,
            self.mujoco_env_frame_skip,
            self.observation_space,
            render_mode=self.render_mode,
            camera_id=self.camera_id
        )

    def _check_model_parameter_dimensions(self):
        '''
        Check if the defined number of variables equals the one from self.model
        
        The number of variables includes the robot, gripper, and objects.
        '''
    
        assert self.ur3_nqpos + self.ball_nqpos == self.model.nq, "# of qpos elements not matching"
        assert self.ur3_nqvel + self.ball_nqvel == self.model.nv, "# of qvel elements not matching"    
        assert self.ur3_nact == self.model.nu, "# of action elements not matching"

    def _define_kinematic_parameters(self):
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
        self.kinematics_params['d'] = np.array([0.1519, 0, 0, 0.11235, 0.08535, 0.0819 + 0.12]) # in m
        self.kinematics_params['a'] = np.array([0, -0.24365, -0.21325, 0, 0, 0]) # in m
        self.kinematics_params['alpha'] =np.array([np.pi/2, 0, 0, np.pi/2, -np.pi/2, 0]) # in rad
        self.kinematics_params['offset'] = np.array([0, 0, 0, 0, 0, 0])  # theta

        # upper and lower bounds for each joint
        self.kinematics_params['ub'] = np.array([2*np.pi for _ in range(6)])
        self.kinematics_params['lb'] = np.array([-2*np.pi for _ in range(6)])

        # transition from world body to base link (right_arm_rotz)
        rotz_idx = mujoco.mj_name2id(self.model, 1, 'right_arm_rotz')

        self.kinematics_params['T'] = np.eye(4)
        self.kinematics_params['T'][0:3,0:3] = self.data.xmat[rotz_idx].reshape([3,3]).copy()
        self.kinematics_params['T'][0:3,3] = self.data.xpos[rotz_idx].copy()

        # pickle the kinematics parameters to the given path if there isn't one already
        path_to_pkl = os.path.join(os.path.dirname(__file__), 'catch_ur3_kinematics_params.pkl')
        if not os.path.isfile(path_to_pkl):
            pickle.dump(self.kinematics_params, open(path_to_pkl, 'wb'))

    def _define_collision_checker_variables(self):
        '''Define new variable pointing to the instance itself to check collision'''

        self.collision_env = self

    def _is_collision(self, qpos):
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

    ##### overriden method (mandatory for reset, step) #####

    def reset_model(self):

        D2R = np.pi / 180

        ur3_qpos = np.array([90, -45, 135, -180, 45, 0]) * D2R
        ur3_qvel = np.zeros(self.ur3_nqvel)

        ball_qpos = np.array([0.4, -2, 1.5, 0, 0, 0, 0])

        ball_trans_vel = np.array([0, 2.5, 2.2]) + 0.2 * np.random.randn(3)
        ball_rot_vel = np.zeros(3)
        ball_qvel = np.concatenate([ball_trans_vel, ball_rot_vel])

        qpos = np.concatenate([ur3_qpos, ball_qpos])
        # mujoco.mj_setState(self.model, self.data, qpos, 1 << 1)

        qvel = np.concatenate([ur3_qvel, ball_qvel])
        # mujoco.mj_setState(self.model, self.data, qvel, 1 << 2)

        self.set_state(qpos, qvel)

        return np.concatenate([qpos, qvel])

    def step(self, action):

        # clip action according to joint torque limiet
        action = np.clip(action, -self.ur3_torque_limit, self.ur3_torque_limit)

        self._step_mujoco_simulation(action, self.mujoco_env_frame_skip)

        observation = self.get_obs()
        reward = self.get_reward()
        terminated = False
        truncated = False
        info = self._get_step_info()

        return observation, reward, terminated, truncated, info

    def get_obs(self):

        qpos = copy(self.data.qpos[:])
        qvel = copy(self.data.qvel[:])

        return np.concatenate([qpos, qvel])
    
    def get_reward(self):

        return 0

    def _get_info(self):

        ur3_qpos = self.data.qpos[:6]

        info = {
            'ur3_qpos': ur3_qpos,
            'ball_pos': self.data.qpos[6:9],
            'time': self.data.time
        }

        return info

    def _get_step_info(self):

        return self._get_info()
    
    def _get_reset_info(self):

        return self._get_info()
    
    ##### ur3 method #####

    def servoj(self, q, a, v, t=0.008, lookahead_time=0.1, gain=300):
        '''
        from URScript API Reference v3.5.4

            q: joint positions (rad)
            a: NOT used in current version
            v: NOT used in current version
            t: time where the command is controlling the robot. The function is blocking for time t [S]
            lookahead_time: time [S], range [0.03,0.2] smoothens the trajectory with this lookahead time
            gain: proportional gain for following target position, range [100,2000]
        '''

        # check given joint coordinates have valid length (6)
        assert q.shape[0] == self.ur3_nqpos
        
        current_q = self._get_ur3_qpos()
        
        # compute p_error, derivative of error, integration over error
        err = q - current_q
        err_dot = - self._get_ur3_qvel()
        self.ur3_err_integ = np.clip(self.ur3_err_integ + err * self.dt, -1, 1)

        # Internal forces
        bias = self._get_ur3_bias()

        # External forces
        constraint = self._get_ur3_constraint()
        constraint = np.clip(constraint, -self.ur3_torque_limit, self.ur3_torque_limit)

        # apply PD control
        gains = [gain, 5, 5]  # PD gains
        ctrl_PD = gains[0] * err + gains[1] * self.ur3_err_integ + gains[2] * err_dot

        return ctrl_PD + bias - constraint

    def _get_ur3_qpos(self):

        return self.data.qpos[:6]

    def _get_ur3_tcp(self):
        
        q = self._get_ur3_qpos()
        _, p, _ = self.forward_kinematics_ee(q)

        return p

    def _get_ur3_qvel(self):
        
        return self.data.qvel[:6]

    def _get_ur3_bias(self):

        return self.data.qfrc_bias[:self.ur3_nqvel]

    def _get_ur3_constraint(self):

        return self.data.qfrc_constraint[0:self.ur3_nqvel]

    ##### utility method #####

    def forward_kinematics_DH(self, q):
        '''
        Compute transition matrix and rotation, translation vectors given joint vector and arm (left/right)
        
        Forward kinematics transforms joint coordinates to cartesian coordinates.
        '''

        assert len(q) == self.ur3_nqpos, "Length of joint coordinates vector q not matching with ur3_nqpos"
        self._define_kinematic_parameters()

        T_0_i = self.kinematics_params['T']
        
        # transition, rotation, translation
        # The matrix and vectors below contains ur3_nqpos + 1 element, each of which from world frame to i-th frame
        T = np.zeros([self.ur3_nqpos+1, 4, 4])
        R = np.zeros([self.ur3_nqpos+1, 3, 3])
        p = np.zeros([self.ur3_nqpos+1, 3])

        # world frame
        T[0,:,:] = T_0_i
        R[0,:,:] = T_0_i[0:3,0:3]
        p[0,:] = T_0_i[0:3,3]

        # from world frame to i-th link body
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
    
    def forward_kinematics_ee(self, q):
        '''Compute forward kinematics to the end effector frame'''

        R, p, T = self.forward_kinematics_DH(q)

        return R[-1, :, :], p[-1, :], T[-1, :]
    
    def _jacobian_DH(self, q):
            
            assert len(q) == self.ur3_nqpos

            epsilon = 1e-6
            epsilon_inv = 1 / epsilon

            _, ps, _ = self.forward_kinematics_DH(q)
            p = ps[-1,:]  # position of end effector with joint coordinates `q`

            jacobian = np.zeros([3, self.ur3_nqpos])

            for i in range(self.ur3_nqpos):

                q_ = q.copy()
                q_[i] = q_[i] + epsilon  # perturb each joint coordinate by epsilon

                _, ps_, _ = self.forward_kinematics_DH(q_)
                p_ = ps_[-1,:]  # perturbed position

                jacobian[:, i] = (p_ - p) * epsilon_inv

            return jacobian

    def inverse_kinematics_ee(self, p_targ, null_obj_func):
        '''Inverse kinematics with forward_kinematics_DH() and _jacobian_DH()'''

        threshold = 0.01
        threshold_null = 0.001
        max_iter = 100
        epsilon = 1e-6

        q = self._get_ur3_qpos  # current joint coordinates
        
        R, p, _ = self.forward_kinematics_ee(q)
        jac = self._jacobian_DH(q)

        delta_p = p_targ - p
        err = np.linalg.norm(delta_p)
        null_obj_val = null_obj_func.evaluate(R)

        iter_taken = 0

        while True:

            if (err < threshold and null_obj_val < threshold_null) or iter_taken >= max_iter: break
            else: iter_taken += 1

            # pseudo-inverse + null-space approach

            jac_dagger = np.linalg.pinv(jac)
            jac_null = np.eye(self.ur3_nqpos) - np.matmul(jac_dagger, jac) # null space of Jacobian

            phi = np.zeros(self.ur3_nqpos) # find phi (null objective derivative)

            for i in range(self.ur3_nqpos):

                q_perturb = q.copy()
                q_perturb[i] += epsilon

                SO3_perturb, _, _ = self.forward_kinematics_ee(q_perturb)
                null_obj_val_perturb = null_obj_func.evaluate(SO3_perturb)

                phi[i] = (null_obj_val_perturb - null_obj_val) / epsilon

            # update
            delta_p = p_targ - p
            delta_q = np.matmul(jac_dagger, p_targ) - np.matmul(jac_null, phi)

            q += delta_q
            q = np.minimum(self.kinematics_params['ub'], np.maximum(q, self.kinematics_params['lb']))

            R, p, _ = self.forward_kinematics_ee(q)
            jac = self._jacobian_DH(q)
            
            null_obj_val = null_obj_func.evaluate(R)
            
            err = np.linalg.norm(delta_p)
        
        if iter_taken == max_iter:
            warnings.warn('Max iteration limit reached! err: %f (threshold: %f), null_obj_err: %f (threshold: %f)'%(err, threshold, null_obj_val, threshold_null),
                RuntimeWarning)
        
        return q, iter_taken, err, null_obj_val

    def inverse_kinematics_ee_align_heading(self, p_target, h_target):
        '''
        Perform numerical inverse kinematics with pseudo-inverse and null objective

        Inputs
        ------
        - `p`
            target position of the end-effector frame origin
        - `h`
            target heading (x-axis) of the end-effector frame

        Return
        ------
        - `q`
            joint angles to achieve target position and heading
        '''

        def null_obj_func(R_current, h_target):
            h_current = R_current @ np.array([1, 0, 0])
            return np.linalg.norm(h_target - h_current)

        threshold = 0.01
        threshold_null = 0.01

        max_iter = 100
        step_size = 0.01

        epsilon = 1e-6

        q_current = self._get_ur3_qpos()  # current joint coordinates
        
        R_current, p_current, _ = self.forward_kinematics_ee(q_current)
        jac_current = self._jacobian_DH(q_current)

        delta_p = p_target - p_current
        err = np.sqrt(np.linalg.norm(delta_p))

        null_obj_val = null_obj_func(R_current, h_target)

        iter_taken = 0

        while True:

            if (err < threshold and null_obj_val < threshold_null) or iter_taken >= max_iter: break
            else: iter_taken += 1

            # pseudo-inverse + null-space approach

            jac_dagger = np.linalg.pinv(jac_current)
            jac_null = np.eye(self.ur3_nqpos) - np.matmul(jac_dagger, jac_current) # null space of Jacobian

            phi = np.zeros(self.ur3_nqpos) # find phi (null objective derivative)

            for i in range(self.ur3_nqpos):

                q_perturb = q_current.copy()
                q_perturb[i] += epsilon

                R_perturb, _, _ = self.forward_kinematics_ee(q_perturb)
                null_obj_val_perturb = null_obj_func(R_perturb, h_target)

                phi[i] = (null_obj_val_perturb - null_obj_val) / epsilon

            # update
            delta_p = p_target - p_current
            delta_q = np.matmul(jac_dagger, step_size * delta_p) - np.matmul(jac_null, step_size * phi)

            q_current += delta_q
            q_current = np.clip(q_current, self.kinematics_params['lb'], self.kinematics_params['ub'])

            R_current, p_current, _ = self.forward_kinematics_ee(q_current)
            jac_current = self._jacobian_DH(q_current)
            
            null_obj_val = null_obj_func(R_current, h_target)
            err = np.linalg.norm(delta_p)

        info = {
            'iter_taken': iter_taken,
            'p_error': err,
            'h_error': null_obj_val
        }
        
        return q_current, info

    def forward_kinematics_ee_pos_eul(self, q):
        '''Return concatenated position and quaternion of end-effector'''

        rotation_matrix, translation_vector, _ = self.forward_kinematics_ee(q)
        rotation = Rotation.from_matrix(rotation_matrix)

        euler_vector = rotation.as_euler('YXZ')

        return np.concatenate([translation_vector, euler_vector])
    
    def inverse_kinematics_ee_pos_eul(self, p_target):

        assert len(p_target) == 6

        # p_target = self.clip_pos_eul(p_target)

        epsilon = 1e-6
        epsilon_inv = 1 / epsilon

        q_current = self._get_ur3_qpos()
        p_current = self.forward_kinematics_ee_pos_eul(q_current)

        jacobian = np.zeros([6, self.ur3_nqpos])

        for i in range(self.ur3_nqpos):

            q_perturbed = q_current.copy()
            q_perturbed[i] = q_perturbed[i] + epsilon  # perturb each joint coordinate by epsilon

            p_perturbed = self.forward_kinematics_ee_pos_eul(q_perturbed)

            p_delta = p_perturbed - p_current
            # p_delta = self.clip_pos_eul(p_delta)

            jacobian[:, i] = p_delta * epsilon_inv

        p_error = p_target - p_current
        q_error = np.linalg.inv(jacobian) @ p_error

        return q_current + q_error

    def clip_pos_eul(self, p):
        
        p_clipped = p.copy()
        p_clipped[3:] = np.mod(p_clipped[3:] + np.pi, 2 * np.pi) - np.pi

        return p_clipped

