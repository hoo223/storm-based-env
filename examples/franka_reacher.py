#
# MIT License
#
# Copyright (c) 2020-2021 NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.#
""" Example spawning a robot in gym 

"""

# --------------------------------------------------- Import Libraries --------------------------------------------------- #

# standard library
import copy
import cv2
import time
import yaml
import argparse

# numpy
import numpy as np
np.set_printoptions(precision=2) # numpy 출력 소수점 자리수 설정 -> 2자리까지만 출력

# matplotlib
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('tkagg') # or 'agg' if you don't have TkAgg installed (e.g. on a server) # matplotlib backend 설정

# isaacgym
from isaacgym import gymapi, gymtorch, gymutil

# torch
import torch
torch.multiprocessing.set_start_method('spawn',force=True) # torch multiprocessing 설정
torch.set_num_threads(8) # torch thread 개수 설정
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# quaternion
from quaternion import quaternion, from_rotation_vector, from_rotation_matrix
from quaternion import from_euler_angles, as_float_array, as_rotation_matrix, from_float_array, as_quat_array

# storm_kit
from storm_kit.gym.core import Gym, World
from storm_kit.gym.sim_robot import RobotSim
from storm_kit.util_file import get_configs_path, get_gym_configs_path, join_path, load_yaml, get_assets_path
from storm_kit.gym.helpers import load_struct_from_dict
from storm_kit.util_file import get_mpc_configs_path as mpc_configs_path
from storm_kit.differentiable_robot_model.coordinate_transform import quaternion_to_matrix, CoordinateTransform
from storm_kit.mpc.task.reacher_task import ReacherTask

# ros
import rospy
from geometry_msgs.msg import Pose, PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

# --------------------------------------------------- Define Functions --------------------------------------------------- #

# define ros node class for teleoperation and publishing joint states, link closest spheres, object closest points and spheres and points for visualization in rviz 
class teleopNode(object):
    def __init__(self, verbose=False, prefix=""):
        self.prefix = prefix
        self.joint_state_pub = rospy.Publisher('joint_states', JointState, queue_size= 10)
        self.link_closest_sphere_pub = rospy.Publisher('link_closest_spheres', Float64MultiArray, queue_size= 10)
        self.object_closest_points_pub = rospy.Publisher('object_closest_points', Float64MultiArray, queue_size= 10)
        self.spheres_and_points_pub = rospy.Publisher('spheres_and_points', Float64MultiArray, queue_size= 10) 
        self.target_pose_sub = rospy.Subscriber(prefix+'/target_pose', PoseStamped, self.target_pose_callback) 
        self.delta_target_input_sub = rospy.Subscriber(prefix+'/delta_target_input', Float64MultiArray, self.delta_target_input_callback)
        self.min_points_sub = rospy.Subscriber('/min_points', Float64MultiArray, self.min_points_callback)
        
        self.target_pose = Pose()
        self.delta_target_input = Float64MultiArray()
        self.min_points = None
    
    # callback functions
    def target_pose_callback(self, data): 
        self.target_pose = data.pose
        #print(self.target_pose.position)
        
    def delta_target_input_callback(self, data):
        self.delta_target_input = data
        #print(self.target_pose.position)
    
    def min_points_callback(self, data):
        self.min_points = data.data
    
    # publish functions
    def publish_joint_states(self, joint_states):
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 
                    'panda_joint5', 'panda_joint6', 'panda_joint7', 
                    'panda_finger_joint1', 'panda_finger_joint2']
        msg.position = list(joint_states['position']) + [0, 0]
        self.joint_state_pub.publish(msg)
        
    def publish_link_closest_spheres(self, spheres):
        msg = Float64MultiArray()
        msg.data = spheres
        self.link_closest_sphere_pub.publish(msg)
        
    def publish_object_closest_points(self, points):
        msg = Float64MultiArray()
        msg.data = points
        self.object_closest_points_pub.publish(msg)
        
    def publish_spheres_and_points(self, spheres, points):
        msg = Float64MultiArray()
        msg.data = spheres + points
        self.spheres_and_points_pub.publish(msg)
        

# define mpc_robot_interactive class
class mpc_robot_interactive(object):
    # define init function
    def __init__(self, args, gym_instance):
        # ros node initialization
        self.tn = teleopNode(prefix='isaac') # teleop node class instantiation 
        
        # set parameters
        self.ee_error = 10.0
        self.q_des = None
        self.qd_des = None
        self.log_traj = {'q':[], 'q_des':[], 'qdd_des':[], 'qd_des':[], 'qddd_des':[]} # trajectory log dictionary
        self.ee_pose = gymapi.Transform()

        # gym and sim instance
        self.gym_instance = gym_instance
        self.gym = gym_instance.gym
        self.sim = gym_instance.sim
        
        # load robot, sim and world parameters from yaml files
        self.vis_ee_target = True
        robot_file = args.robot + '.yml'
        task_file = args.robot + '_reacher.yml'
        world_file = 'collision_primitives_3d.yml'
        
        world_yml = join_path(get_gym_configs_path(), world_file)
        with open(world_yml) as file:
            world_params = yaml.load(file, Loader=yaml.FullLoader)

        robot_yml = join_path(get_gym_configs_path(),args.robot + '.yml')

        with open(robot_yml) as file:
            robot_params = yaml.load(file, Loader=yaml.FullLoader)
        sim_params = robot_params['sim_params']
        sim_params['asset_root'] = get_assets_path()
        sim_params['collision_model'] = None
        
        # set device to cpu or gpu depending on args
        if(args.cuda):
            device = 'cuda'
        else:
            device = 'cpu'
        
        # create robot simulation:
        self.robot_sim = RobotSim(gym_instance=self.gym, sim_instance=self.sim, **sim_params, device=device)
        
        # create gym environment:
        robot_pose = sim_params['robot_pose']
        self.env_ptr = gym_instance.env_list[0]
        self.robot_ptr = self.robot_sim.spawn_robot(self.env_ptr, robot_pose, coll_id=2)

        # spawn camera:
        # robot_camera_pose = np.array([1.6,-1.5, 1.8,0.707,0.0,0.0,0.707])
        # q = as_float_array(from_euler_angles(-0.5 * 90.0 * 0.01745, 50.0 * 0.01745, 90 * 0.01745))
        # robot_camera_pose[3:] = np.array([q[1], q[2], q[3], q[0]])
        robot_camera_pose = np.array([1.6,-1.5, 1.8,0.707,0.0,0.0,0.707])
        self.assist_cam_handle = self.robot_sim.spawn_camera(self.env_ptr, 60, 640, 480, robot_camera_pose)

        # get pose of robot in world frame
        self.w_T_r = copy.deepcopy(self.robot_sim.spawn_robot_pose)
        w_T_robot = torch.eye(4)
        quat = torch.tensor([self.w_T_r.r.w, self.w_T_r.r.x, self.w_T_r.r.y, self.w_T_r.r.z]).unsqueeze(0)
        rot = quaternion_to_matrix(quat)
        w_T_robot[0,3] = self.w_T_r.p.x
        w_T_robot[1,3] = self.w_T_r.p.y
        w_T_robot[2,3] = self.w_T_r.p.z
        w_T_robot[:3,:3] = rot[0]
        
        # create world instance
        self.world_instance = World(self.gym, self.sim, self.env_ptr, world_params, w_T_r=self.w_T_r)
        
        # table_dims = np.ravel([1.5,2.5,0.7])
        # cube_pose = np.ravel([0.35, -0.0,-0.35,0.0, 0.0, 0.0,1.0])

        # cube_pose = np.ravel([0.9,0.3,0.4, 0.0, 0.0, 0.0,1.0])
        # table_dims = np.ravel([0.35,0.1,0.8])
        
        # cube_pose = np.ravel([0.35,0.3,0.4, 0.0, 0.0, 0.0,1.0])
        # table_dims = np.ravel([0.3,0.1,0.8])
        
        
        device = torch.device('cuda', 0) # set device to gpu
        tensor_args = {'device':device, 'dtype':torch.float32} # set tensor args 

        # MPC class instantiation
        self.mpc_control = ReacherTask(task_file, robot_file, world_file, tensor_args)
        
        # distance_checker 
        self.robot_world_coll = self.mpc_control.controller.rollout_fn.primitive_collision_cost.robot_world_coll
        self.link_pos_batch = torch.zeros([1, 6, 3], **tensor_args)
        self.link_rot_batch = torch.zeros([1, 6, 3, 3], **tensor_args)
        
        # load robot dof 
        self.n_dof = self.mpc_control.controller.rollout_fn.dynamics_model.n_dofs

        # initial acc
        start_qdd = torch.zeros(self.n_dof, **tensor_args)

        # load experiment parameters
        exp_params = self.mpc_control.exp_params
        
        # update goal:
        franka_bl_state = np.array([-0.3, 0.3, 0.2, -2.0, 0.0, 2.4, 0.0,
                                    0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0])
        x_des_list = [franka_bl_state]
        x_des = x_des_list[0]
        self.mpc_control.update_params(goal_state=x_des)

        # spawn object:
        x,y,z = 0.0, 0.0, 0.0
        tray_color = gymapi.Vec3(0.8, 0.1, 0.1)
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.001
        asset_options.fix_base_link = True
        asset_options.thickness = 0.002

        self.object_pose = gymapi.Transform()
        self.object_pose.p = gymapi.Vec3(x, y, z)
        self.object_pose.r = gymapi.Quat(0, 0, 0, 1)
        
        if(self.vis_ee_target):
            # mug attached to target pose
            obj_asset_file = "urdf/mug/movable_mug.urdf" 
            obj_asset_root = get_assets_path()  
            target_object = self.world_instance.spawn_object(obj_asset_file, obj_asset_root, self.object_pose, color=tray_color, name='ee_target_object') # print(target_object) -> 5
            self.obj_base_handle = self.gym.get_actor_rigid_body_handle(self.env_ptr, target_object, 0) # print(self.obj_base_handle) -> 12
            self.obj_body_handle = self.gym.get_actor_rigid_body_handle(self.env_ptr, target_object, 6) # print(self.obj_body_handle) -> 18
            self.gym.set_rigid_body_color(self.env_ptr, target_object, 0, gymapi.MESH_VISUAL_AND_COLLISION, tray_color)
            self.gym.set_rigid_body_color(self.env_ptr, target_object, 6, gymapi.MESH_VISUAL_AND_COLLISION, tray_color)

            # mug attached to ee
            obj_asset_file = "urdf/mug/mug.urdf"
            obj_asset_root = get_assets_path()
            self.ee_handle = self.world_instance.spawn_object(obj_asset_file, obj_asset_root, self.object_pose, color=tray_color, name='ee_current_as_mug')
            self.ee_body_handle = self.gym.get_actor_rigid_body_handle(self.env_ptr, self.ee_handle, 0)
            tray_color = gymapi.Vec3(0.0, 0.8, 0.0)
            self.gym.set_rigid_body_color(self.env_ptr, self.ee_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, tray_color)

        # set target object pose
        self.g_pos = np.ravel(self.mpc_control.controller.rollout_fn.goal_ee_pos.cpu().numpy()) # goal position
        self.g_q = np.ravel(self.mpc_control.controller.rollout_fn.goal_ee_quat.cpu().numpy()) # goal orientation
        self.object_pose.p = gymapi.Vec3(self.g_pos[0], self.g_pos[1], self.g_pos[2]) # set goal position 
        self.object_pose.r = gymapi.Quat(self.g_q[1], self.g_q[2], self.g_q[3], self.g_q[0]) # set goal orientation
        self.object_pose = self.w_T_r * self.object_pose # transform to world frame
        if(self.vis_ee_target):
            self.gym.set_rigid_transform(self.env_ptr, self.obj_base_handle, self.object_pose)

        # get transform from world to robot
        self.w_robot_coord = CoordinateTransform(trans=w_T_robot[0:3,3].unsqueeze(0),
                                            rot=w_T_robot[0:3,0:3].unsqueeze(0))

        self.rollout = self.mpc_control.controller.rollout_fn # rollout_fn = ArmReacher
        mpc_tensor_dtype = {'device':device, 'dtype':torch.float32}
        self.tensor_args = mpc_tensor_dtype

        self.sim_dt = self.mpc_control.exp_params['control_dt'] # 0.01
        
        # # assistant camera
        # assist_cam_props = gymapi.CameraProperties()
        # assist_cam_props.width = 128
        # assist_cam_props.height = 128
        # assist_cam_handle = gym.create_camera_sensor(env_ptr, assist_cam_props)
        # gym.set_camera_location(assist_cam_handle, env_ptr, gymapi.Vec3(1,1,1), gymapi.Vec3(0,0,0))

        self.t_step = gym_instance.get_sim_time() # get current sim time        
        
    # define function for updating goal pose
    def update_goal_pose(self):
        # if(self.vis_ee_target):
        #     pose = copy.deepcopy(self.world_instance.get_pose(self.obj_body_handle)) # get pose of target object
        #     pose = copy.deepcopy(self.w_T_r.inverse() * pose) # transform to robot frame
        #     # update goal position and orientation if they have changed 
        #     if(np.linalg.norm(self.g_pos - np.ravel([pose.p.x, pose.p.y, pose.p.z])) > 0.00001 or (np.linalg.norm(self.g_q - np.ravel([pose.r.w, pose.r.x, pose.r.y, pose.r.z]))>0.0)):
        #         self.g_pos = np.array([pose.p.x, pose.p.y, pose.p.z])
        #         self.g_q = np.array([pose.r.w, pose.r.x, pose.r.y, pose.r.z])
        #         self.mpc_control.update_params(goal_ee_pos=self.g_pos, goal_ee_quat=self.g_q) # update goal position and orientation
        
        # update goal pose
        self.g_pos += np.array([self.control_vec_p.x, self.control_vec_p.y, self.control_vec_p.z])*5
        self.g_q = np.ravel(self.mpc_control.controller.rollout_fn.goal_ee_quat.cpu().numpy())
        self.mpc_control.update_params(goal_ee_pos=self.g_pos, goal_ee_quat=self.g_q)
        # update visualization of goal pose
        self.object_pose.p = gymapi.Vec3(self.g_pos[0], self.g_pos[1], self.g_pos[2])
        self.object_pose.r = gymapi.Quat(self.g_q[1], self.g_q[2], self.g_q[3], self.g_q[0])
        self.object_pose = self.w_T_r * self.object_pose # 왜 하는지 알아보기
        if(self.vis_ee_target):
            self.gym.set_rigid_transform(self.env_ptr, self.obj_base_handle, self.object_pose)
                
    # define function for updating control vector
    def update_control_vector(self):
        self.control_vec_p = gymapi.Vec3(self.tn.delta_target_input.data[0], self.tn.delta_target_input.data[1], self.tn.delta_target_input.data[2])
        self.control_vec_r = gymapi.Vec3(self.tn.delta_target_input.data[3], self.tn.delta_target_input.data[4], self.tn.delta_target_input.data[5])
        
        # # Get viewer camera pose
        # self.vcam_pose = self.gym.get_viewer_camera_transform(self.gym_instance.viewer, self.env_ptr) # with respect to the selected environment
        # self.vcam_left = self.vcam_pose.r.rotate(gymapi.Vec3(1, 0, 0)) # 카메라 좌표계의 x축 계산 (env frame 기준) : (-1, 0, 0)
        # self.vcam_up = self.vcam_pose.r.rotate(gymapi.Vec3(0, 1, 0)) # 카메라 좌표계의 y축 계산 (env frame 기준) : (0, 0, 1)
        # self.vcam_fwd = self.vcam_pose.r.rotate(gymapi.Vec3(0, 0, 1)) # 카메라 좌표계의 z축 계산 (env frame 기준) : (0, 1, 0)
        
        # # Control vector from control frame to cam frame
        # self.vcam_fwd.z = 0
        # self.vcam_left.z = 0
        # self.control_vec_cam = self.vcam_fwd * self.control_vec_p.x + self.vcam_left * self.control_vec_p.y + self.vcam_up * self.control_vec_p.z
        
    # define function for getting current state and pose
    def get_current_state_and_pose(self):
        self.current_robot_state = copy.deepcopy(self.robot_sim.get_state(self.env_ptr, self.robot_ptr)) # dictionary
        curr_state = np.hstack((self.current_robot_state['position'], self.current_robot_state['velocity'], self.current_robot_state['acceleration'])) # 7 x 3 = 21
        self.curr_state_tensor = torch.as_tensor(curr_state, **self.tensor_args).unsqueeze(0) # torch.Size([1, 21])
        
    # define function for updating command from mpc
    def update_command_from_mpc(self):
        self.command = self.mpc_control.get_command(self.t_step, self.current_robot_state, control_dt=self.sim_dt, WAIT=True)
        self.q_des = copy.deepcopy(self.command['position']) # get position command: (7,)
        #self.qd_des = copy.deepcopy(command['velocity']) #* 0.5 # get velocity command: (7,)
        #self.qdd_des = copy.deepcopy(command['acceleration']) # get acceleration command: (7,)
          
    # define function for updating ee mug pose
    def update_ee_mug_pose(self):
        # get current ee pose:
        pose_state = self.mpc_control.controller.rollout_fn.get_ee_pose(self.curr_state_tensor) # get command 이후에 해야함
        e_pos = np.ravel(pose_state['ee_pos_seq'].cpu().numpy())
        e_quat = np.ravel(pose_state['ee_quat_seq'].cpu().numpy())
        self.ee_pose.p = copy.deepcopy(gymapi.Vec3(e_pos[0], e_pos[1], e_pos[2]))
        self.ee_pose.r = gymapi.Quat(e_quat[1], e_quat[2], e_quat[3], e_quat[0])
        self.ee_pose = copy.deepcopy(self.w_T_r) * copy.deepcopy(self.ee_pose) # transform to world frame
        
        # update visualization of ee mug
        if(self.vis_ee_target):
            self.gym.set_rigid_transform(self.env_ptr, self.ee_body_handle, copy.deepcopy(self.ee_pose))
    
    # define function for trajectory visualization
    def visualize_trajectory(self):
        # sample trajectory visualization
        self.gym_instance.clear_lines()
        top_trajs = self.mpc_control.top_trajs.cpu().float()#.numpy() # control_process.top_trajs
        n_p, n_t = top_trajs.shape[0], top_trajs.shape[1]
        w_pts = self.w_robot_coord.transform_point(top_trajs.view(n_p * n_t, 3)).view(n_p, n_t, 3)

        top_trajs = w_pts.cpu().numpy() # <class 'numpy.ndarray'> (10, 30, 3)
        color = np.array([0.0, 1.0, 0.0])
        for k in range(top_trajs.shape[0]):
            pts = top_trajs[k,:,:]
            color[0] = float(k) / float(top_trajs.shape[0])
            color[1] = 1.0 - float(k) / float(top_trajs.shape[0])
            self.gym_instance.draw_lines(pts, color=color)
            
    # define function for checking distance between robot and world
    def check_distance(self):
        #print(rollout.dynamics_model.robot_model.get_link_pose('panda_hand'))
        for ki, k in enumerate(['panda_link2','panda_link3','panda_link4','panda_link5', 'panda_link6','panda_hand']):
            #print(ki, k)
            link_pos, link_rot = self.rollout.dynamics_model.robot_model.get_link_pose(k)
            self.link_pos_batch[0, ki] = link_pos
            self.link_rot_batch[0, ki] = link_rot
        
        self.min_dist = self.robot_world_coll.check_robot_sphere_collisions(self.link_pos_batch, self.link_rot_batch) # 링크를 구성하는 sphere 중 환경 물체와의 거리가 가장 가까운 sphere의 거리를 반환
        self.link_closest_spheres = self.robot_world_coll.link_closest_spheres # 각 링크에서 환경과 가장 가까운 sphere 정보
        self.object_closest_points  = self.robot_world_coll.object_closest_points # 각 링크와 가장 가까운 물체의 정보
        #p_closest = robot_world_coll.world_coll.p_closest # 각 그리드에서 가장 가까운 물체 상의 점 [38400, 4, 1], 4 : 1 sphere + 3 cubes
        #print(p_closest[0, :, 0])
        #print(min_dist, link_closest_sphere)
        
    # define function for publishing info to ROS
    def publish_info(self):
        joint_states = self.robot_sim.get_state(self.env_ptr, self.robot_ptr) # get joint states
        self.tn.publish_joint_states(joint_states)
        self.tn.publish_spheres_and_points(self.link_closest_spheres, self.object_closest_points)
        
        #tn.publish_link_closest_spheres(link_closest_spheres) # 링크 상 sphere 정보만 보내고 싶을 때
        #tn.publish_object_closest_points(object_closest_points) # 물체 상 sphere 정보만 보내고 싶을 때
        
    # define function for updating assistive camera
    def update_assistive_camera(self):
        if self.tn.min_points is not None:
            print(self.tn.min_points)
            
            # obtain camera info
            camera_data = self.robot_sim.observe_camera(self.env_ptr)
            
            # change assistant cam pose
            cam_pose = camera_data['robot_camera_pose']
            cam_pos = cam_pose[:3]
            cam_rot = cam_pose[3:]
            #print(cam_pose)
        
            #if i%50 == 0:
            self.gym.set_camera_location(self.assist_cam_handle, self.env_ptr, gymapi.Vec3(1.6,1.5, 1.8), gymapi.Vec3(0,0,0)) # y-axis -> UP # 회전은 왠지 카메라 좌표계 기준 느낌
            
            # show assistant cam
            try:
                img = camera_data['color']
                cv2.imshow('w', img)
                cv2.waitKey(1)
            except:
                print('no image')
        
    # define function for mpc loop
    def mpc_loop(self):
        i = 0
        while not self.gym.query_viewer_has_closed(self.gym_instance.viewer): # while the viewer is open
            try:
                self.gym_instance.step() # step the simulation
                self.t_step += self.sim_dt # update current sim time
                
                # update control vector by teleoperation
                self.update_control_vector() # out: self.control_vec_p, self.control_vec_r
                
                # update goal pose
                self.update_goal_pose() # in: self.control_vec_p, self.control_vec_r
                                        # out: self.object_pose
                
                # get current state and pose
                self.get_current_state_and_pose() # out: self.current_robot_state, self.curr_state_tensor
                
                # get command from mpc
                self.get_command_from_mpc() # in: self.t_step, self.current_robot_state
                                            # out: self.command, self.q_des
                
                # update ee mug pose
                self.update_ee_mug_pose() # in: self.curr_state_tensor
                                          # out: self.ee_pose

                # trajectory visualization
                self.visualize_trajectory()
                
                # send position command to sim
                self.robot_sim.command_robot_position(self.q_des, self.env_ptr, self.robot_ptr)
                #self.robot_sim.set_robot_state(q_des, qd_des, env_ptr, robot_ptr)
                
                # check distance between robot and world
                self.check_distance() # out: self.min_dist, self.link_closest_spheres, self.object_closest_points
                
                # publish info to visualization node
                self.publish_info() # in: self.link_closest_spheres, self.object_closest_points
                
                # assistive camera
                #self.update_assistive_camera()
                
                # check error
                #ee_error = self.mpc_control.get_current_error(current_robot_state)
                #print(["{:.3f}".format(x) for x in ee_error], "{:.3f}".format(mpc_control.opt_dt), "{:.3f}".format(mpc_control.mpc_dt))
                
                # update index
                i += 1
                
            except KeyboardInterrupt:
                print('Closing')
                done = True
                break
            
        self.mpc_control.close()
        return 1 
    
if __name__ == '__main__':
    
    # instantiate empty gym:
    parser = argparse.ArgumentParser(description='pass args')
    parser.add_argument('--robot', type=str, default='franka', help='Robot to spawn')
    parser.add_argument('--cuda', action='store_true', default=True, help='use cuda')
    parser.add_argument('--headless', action='store_true', default=False, help='headless gym')
    parser.add_argument('--control_space', type=str, default='acc', help='Robot to spawn')
    args = parser.parse_args()
    
    sim_params = load_yaml(join_path(get_gym_configs_path(),'physx.yml')) # /home/lhs/storm/content/configs/gym/physx.yml 
    sim_params['headless'] = args.headless
    gym_instance = Gym(**sim_params)
    
    # Node initialization
    rospy.init_node("storm", anonymous=True) # storm_node 
    
    mpc_class = mpc_robot_interactive(args, gym_instance)
    mpc_class.mpc_loop() # mpc loop