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
        
# define function for mpc loop
def mpc_robot_interactive(args, gym_instance):
    # ros node initialization
    tn = teleopNode(prefix='isaac') # teleop node class instantiation 
    
    # set parameters
    ee_error = 10.0
    i = 0
    q_des = None
    qd_des = None
    log_traj = {'q':[], 'q_des':[], 'qdd_des':[], 'qd_des':[], 'qddd_des':[]} # trajectory log dictionary
    ee_pose = gymapi.Transform()

    # gym and sim instance
    gym = gym_instance.gym
    sim = gym_instance.sim
    
    # load robot, sim and world parameters from yaml files
    vis_ee_target = True
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
    robot_sim = RobotSim(gym_instance=gym, sim_instance=sim, **sim_params, device=device)
    
    # create gym environment:
    robot_pose = sim_params['robot_pose']
    env_ptr = gym_instance.env_list[0]
    robot_ptr = robot_sim.spawn_robot(env_ptr, robot_pose, coll_id=2)

    # spawn camera:
    # robot_camera_pose = np.array([1.6,-1.5, 1.8,0.707,0.0,0.0,0.707])
    # q = as_float_array(from_euler_angles(-0.5 * 90.0 * 0.01745, 50.0 * 0.01745, 90 * 0.01745))
    # robot_camera_pose[3:] = np.array([q[1], q[2], q[3], q[0]])
    robot_camera_pose = np.array([1.6,-1.5, 1.8,0.707,0.0,0.0,0.707])
    assist_cam_handle = robot_sim.spawn_camera(env_ptr, 60, 640, 480, robot_camera_pose)

    # get pose of robot in world frame
    w_T_r = copy.deepcopy(robot_sim.spawn_robot_pose)
    w_T_robot = torch.eye(4)
    quat = torch.tensor([w_T_r.r.w,w_T_r.r.x,w_T_r.r.y,w_T_r.r.z]).unsqueeze(0)
    rot = quaternion_to_matrix(quat)
    w_T_robot[0,3] = w_T_r.p.x
    w_T_robot[1,3] = w_T_r.p.y
    w_T_robot[2,3] = w_T_r.p.z
    w_T_robot[:3,:3] = rot[0]
    
    # create world instance
    world_instance = World(gym, sim, env_ptr, world_params, w_T_r=w_T_r)
    
    # table_dims = np.ravel([1.5,2.5,0.7])
    # cube_pose = np.ravel([0.35, -0.0,-0.35,0.0, 0.0, 0.0,1.0])

    # cube_pose = np.ravel([0.9,0.3,0.4, 0.0, 0.0, 0.0,1.0])
    # table_dims = np.ravel([0.35,0.1,0.8])
    
    # cube_pose = np.ravel([0.35,0.3,0.4, 0.0, 0.0, 0.0,1.0])
    # table_dims = np.ravel([0.3,0.1,0.8])
    
    
    device = torch.device('cuda', 0) # set device to gpu
    tensor_args = {'device':device, 'dtype':torch.float32} # set tensor args 

    # MPC class instantiation
    mpc_control = ReacherTask(task_file, robot_file, world_file, tensor_args)
    
    # distance_checker 
    robot_world_coll = mpc_control.controller.rollout_fn.primitive_collision_cost.robot_world_coll
    link_pos_batch = torch.zeros([1, 6, 3], **tensor_args)
    link_rot_batch = torch.zeros([1, 6, 3, 3], **tensor_args)
    
    # load robot dof 
    n_dof = mpc_control.controller.rollout_fn.dynamics_model.n_dofs

    # initial acc
    start_qdd = torch.zeros(n_dof, **tensor_args)

    # load experiment parameters
    exp_params = mpc_control.exp_params
    
    # update goal:
    franka_bl_state = np.array([-0.3, 0.3, 0.2, -2.0, 0.0, 2.4, 0.0,
                                 0.0, 0.0, 0.0,  0.0, 0.0, 0.0, 0.0])
    x_des_list = [franka_bl_state]
    x_des = x_des_list[0]
    mpc_control.update_params(goal_state=x_des)

    # spawn object:
    x,y,z = 0.0, 0.0, 0.0
    tray_color = gymapi.Vec3(0.8, 0.1, 0.1)
    asset_options = gymapi.AssetOptions()
    asset_options.armature = 0.001
    asset_options.fix_base_link = True
    asset_options.thickness = 0.002

    object_pose = gymapi.Transform()
    object_pose.p = gymapi.Vec3(x, y, z)
    object_pose.r = gymapi.Quat(0, 0, 0, 1)
    
    obj_asset_file = "urdf/mug/movable_mug.urdf" 
    obj_asset_root = get_assets_path()
    
    if(vis_ee_target):
        target_object = world_instance.spawn_object(obj_asset_file, obj_asset_root, object_pose, color=tray_color, name='ee_target_object')
        obj_base_handle = gym.get_actor_rigid_body_handle(env_ptr, target_object, 0)
        obj_body_handle = gym.get_actor_rigid_body_handle(env_ptr, target_object, 6)
        gym.set_rigid_body_color(env_ptr, target_object, 0, gymapi.MESH_VISUAL_AND_COLLISION, tray_color)
        gym.set_rigid_body_color(env_ptr, target_object, 6, gymapi.MESH_VISUAL_AND_COLLISION, tray_color)

        obj_asset_file = "urdf/mug/mug.urdf"
        obj_asset_root = get_assets_path()

        ee_handle = world_instance.spawn_object(obj_asset_file, obj_asset_root, object_pose, color=tray_color, name='ee_current_as_mug')
        ee_body_handle = gym.get_actor_rigid_body_handle(env_ptr, ee_handle, 0)
        tray_color = gymapi.Vec3(0.0, 0.8, 0.0)
        gym.set_rigid_body_color(env_ptr, ee_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, tray_color)

    # set target object pose
    g_pos = np.ravel(mpc_control.controller.rollout_fn.goal_ee_pos.cpu().numpy()) # goal position
    g_q = np.ravel(mpc_control.controller.rollout_fn.goal_ee_quat.cpu().numpy()) # goal orientation
    object_pose.p = gymapi.Vec3(g_pos[0], g_pos[1], g_pos[2]) # set goal position 
    object_pose.r = gymapi.Quat(g_q[1], g_q[2], g_q[3], g_q[0]) # set goal orientation
    object_pose = w_T_r * object_pose # transform to world frame
    if(vis_ee_target):
        gym.set_rigid_transform(env_ptr, obj_base_handle, object_pose)

    # get transform from world to robot
    w_robot_coord = CoordinateTransform(trans=w_T_robot[0:3,3].unsqueeze(0),
                                        rot=w_T_robot[0:3,0:3].unsqueeze(0))

    rollout = mpc_control.controller.rollout_fn # rollout_fn = ArmReacher
    mpc_tensor_dtype = {'device':device, 'dtype':torch.float32}
    tensor_args = mpc_tensor_dtype

    sim_dt = mpc_control.exp_params['control_dt'] # 0.01
    
    # # assistant camera
    # assist_cam_props = gymapi.CameraProperties()
    # assist_cam_props.width = 128
    # assist_cam_props.height = 128
    # assist_cam_handle = gym.create_camera_sensor(env_ptr, assist_cam_props)
    # gym.set_camera_location(assist_cam_handle, env_ptr, gymapi.Vec3(1,1,1), gymapi.Vec3(0,0,0))

    t_step = gym_instance.get_sim_time() # get current sim time    

    while not gym.query_viewer_has_closed(gym_instance.viewer): # while the viewer is open
        try:
            gym_instance.step() # step the simulation
            
            if(vis_ee_target):
                pose = copy.deepcopy(world_instance.get_pose(obj_body_handle)) # get pose of target object
                pose = copy.deepcopy(w_T_r.inverse() * pose)

                if(np.linalg.norm(g_pos - np.ravel([pose.p.x, pose.p.y, pose.p.z])) > 0.00001 or (np.linalg.norm(g_q - np.ravel([pose.r.w, pose.r.x, pose.r.y, pose.r.z]))>0.0)):
                    g_pos[0] = pose.p.x
                    g_pos[1] = pose.p.y
                    g_pos[2] = pose.p.z
                    g_q[1] = pose.r.x
                    g_q[2] = pose.r.y
                    g_q[3] = pose.r.z
                    g_q[0] = pose.r.w
                    mpc_control.update_params(goal_ee_pos=g_pos, goal_ee_quat=g_q) # update goal position and orientation
            
            t_step += sim_dt # update current sim time
            
            
            # Get viewer camera pose
            cam_pose = gym.get_viewer_camera_transform(gym_instance.viewer, env_ptr) # with respect to the selected environment
            cam_left = cam_pose.r.rotate(gymapi.Vec3(1, 0, 0)) # 카메라 좌표계의 x축 계산 (env frame 기준) : (-1, 0, 0)
            cam_up = cam_pose.r.rotate(gymapi.Vec3(0, 1, 0)) # 카메라 좌표계의 y축 계산 (env frame 기준) : (0, 0, 1)
            cam_fwd = cam_pose.r.rotate(gymapi.Vec3(0, 0, 1)) # 카메라 좌표계의 z축 계산 (env frame 기준) : (0, 1, 0)
            
            # Control vector in control frame
            control_vec_p = gymapi.Vec3(tn.delta_target_input.data[0], tn.delta_target_input.data[1], tn.delta_target_input.data[2])
            control_vec_r = gymapi.Vec3(tn.delta_target_input.data[3], tn.delta_target_input.data[4], tn.delta_target_input.data[5])
            
            # Control vector from control frame to cam frame
            cam_fwd.z = 0
            cam_left.z = 0
            control_vec_cam = cam_fwd * control_vec_p.x + cam_left * control_vec_p.y + cam_up * control_vec_p.z
            g_pos += np.array([control_vec_p.x, control_vec_p.y, control_vec_p.z])*5
            
            # update goal_ee_pos
            mpc_control.update_params(goal_ee_pos=g_pos)
            g_q = np.ravel(mpc_control.controller.rollout_fn.goal_ee_quat.cpu().numpy())
            object_pose.p = gymapi.Vec3(g_pos[0], g_pos[1], g_pos[2])
            object_pose.r = gymapi.Quat(g_q[1], g_q[2], g_q[3], g_q[0])
            object_pose = w_T_r * object_pose # 왜 하는지 알아보기
            if(vis_ee_target):
                gym.set_rigid_transform(env_ptr, obj_base_handle, object_pose)
            
            current_robot_state = copy.deepcopy(robot_sim.get_state(env_ptr, robot_ptr))
            
            
            command = mpc_control.get_command(t_step, current_robot_state, control_dt=sim_dt, WAIT=True)

            filtered_state_mpc = current_robot_state #mpc_control.current_state
            curr_state = np.hstack((filtered_state_mpc['position'], filtered_state_mpc['velocity'], filtered_state_mpc['acceleration']))

            curr_state_tensor = torch.as_tensor(curr_state, **tensor_args).unsqueeze(0)
            # get position command:
            q_des = copy.deepcopy(command['position'])
            qd_des = copy.deepcopy(command['velocity']) #* 0.5
            qdd_des = copy.deepcopy(command['acceleration'])
            
            ee_error = mpc_control.get_current_error(filtered_state_mpc)
             
            pose_state = mpc_control.controller.rollout_fn.get_ee_pose(curr_state_tensor)
            
            # get current pose:
            e_pos = np.ravel(pose_state['ee_pos_seq'].cpu().numpy())
            e_quat = np.ravel(pose_state['ee_quat_seq'].cpu().numpy())
            ee_pose.p = copy.deepcopy(gymapi.Vec3(e_pos[0], e_pos[1], e_pos[2]))
            ee_pose.r = gymapi.Quat(e_quat[1], e_quat[2], e_quat[3], e_quat[0])
            
            ee_pose = copy.deepcopy(w_T_r) * copy.deepcopy(ee_pose)
            
            if(vis_ee_target):
                gym.set_rigid_transform(env_ptr, ee_body_handle, copy.deepcopy(ee_pose))

            #print(["{:.3f}".format(x) for x in ee_error], "{:.3f}".format(mpc_control.opt_dt), "{:.3f}".format(mpc_control.mpc_dt))
        
            # sample trajectory visualization
            gym_instance.clear_lines()
            top_trajs = mpc_control.top_trajs.cpu().float()#.numpy() # control_process.top_trajs
            n_p, n_t = top_trajs.shape[0], top_trajs.shape[1]
            w_pts = w_robot_coord.transform_point(top_trajs.view(n_p * n_t, 3)).view(n_p, n_t, 3)


            top_trajs = w_pts.cpu().numpy() # <class 'numpy.ndarray'> (10, 30, 3)
            color = np.array([0.0, 1.0, 0.0])
            for k in range(top_trajs.shape[0]):
                pts = top_trajs[k,:,:]
                color[0] = float(k) / float(top_trajs.shape[0])
                color[1] = 1.0 - float(k) / float(top_trajs.shape[0])
                gym_instance.draw_lines(pts, color=color)
            
            # send position command to sim
            robot_sim.command_robot_position(q_des, env_ptr, robot_ptr)
            #robot_sim.set_robot_state(q_des, qd_des, env_ptr, robot_ptr)
            current_state = command
            
            # check distance
            #print(rollout.dynamics_model.robot_model.get_link_pose('panda_hand'))
            for ki, k in enumerate(['panda_link2','panda_link3','panda_link4','panda_link5', 'panda_link6','panda_hand']):
                #print(ki, k)
                link_pos, link_rot = rollout.dynamics_model.robot_model.get_link_pose(k)
                link_pos_batch[0, ki] = link_pos
                link_rot_batch[0, ki] = link_rot
            
            dist = robot_world_coll.check_robot_sphere_collisions(link_pos_batch, link_rot_batch)
            link_closest_spheres = robot_world_coll.link_closest_spheres
            object_closest_points  = robot_world_coll.object_closest_points
            #p_closest = robot_world_coll.world_coll.p_closest # 각 그리드에서 가장 가까운 물체 상의 점 [38400, 4, 1], 4 : 1 sphere + 3 cubes
            #print(p_closest[0, :, 0])
            #print(dist, link_closest_sphere)
            
            # publish current joint states
            #dof_states = robot_sim.gym.get_actor_dof_states(env_ptr, robot_ptr, gymapi.STATE_ALL)
            joint_states = robot_sim.get_state(env_ptr, robot_ptr)
            #print(joint_states)
            tn.publish_joint_states(joint_states)
            #tn.publish_link_closest_spheres(link_closest_spheres)
            #tn.publish_object_closest_points(object_closest_points)
            tn.publish_spheres_and_points(link_closest_spheres, object_closest_points)
            #print(link_closest_spheres, object_closest_points)
            
            
            if tn.min_points is not None:
                print(tn.min_points)
                
                # obtain camera info
                camera_data = robot_sim.observe_camera(env_ptr)
                
                # change assistant cam pose
                cam_pose = camera_data['robot_camera_pose']
                cam_pos = cam_pose[:3]
                cam_rot = cam_pose[3:]
                #print(cam_pose)
            
                #if i%50 == 0:
                gym.set_camera_location(assist_cam_handle, env_ptr, gymapi.Vec3(1.6,1.5, 1.8), gymapi.Vec3(0,0,0)) # y-axis -> UP # 회전은 왠지 카메라 좌표계 기준 느낌
                
                # show assistant cam
                try:
                    img = camera_data['color']
                    cv2.imshow('w', img)
                    cv2.waitKey(1)
                except:
                    print('no image')
            
            # update index
            i += 1
            
        except KeyboardInterrupt:
            print('Closing')
            done = True
            break
    mpc_control.close()
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
    rospy.init_node("storm", anonymous=True)
    
    mpc_robot_interactive(args, gym_instance)