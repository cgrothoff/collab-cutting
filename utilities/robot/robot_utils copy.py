import numpy as np
import traceback
import json
import time
import yaml
import interval
import math
from scipy.signal import butter
from copy import copy
import rospy, actionlib
from joystick import Joystick
from trajectory import Trajectory
from urdf_parser_py.urdf import URDF
from pykdl_utils.kdl_kinematics import KDLKinematics
from tf.transformations import euler_from_matrix, euler_from_quaternion, quaternion_from_euler
from collections import deque
from std_msgs.msg import Float64MultiArray, Float32MultiArray, String
from geometry_msgs.msg import WrenchStamped, PoseStamped, TwistStamped, Wrench, Pose
import copy
from controller_manager_msgs.srv import (
    SwitchController, SwitchControllerRequest, SwitchControllerResponse,
    ListControllers, ListControllersRequest, ListControllersResponse,
    LoadController, LoadControllerRequest, LoadControllerResponse,
    UnloadController, UnloadControllerRequest, UnloadControllerResponse
)
from control_msgs.msg import (
    FollowJointTrajectoryAction,
    FollowJointTrajectoryGoal,
)
from trajectory_msgs.msg import JointTrajectoryPoint
from sensor_msgs.msg import JointState
from geometry_msgs.msg import(
    TwistStamped, Twist,
    WrenchStamped, Wrench,
    PoseStamped, Pose
)
import dynamic_reconfigure.client


LIMIT_X = [-0.3, 0.3]
LIMIT_Y = [-0.9, -0.3]
LIMIT_Z = [0.18, 0.4]
def constrain_wayp(waypoints):
    
    assert LIMIT_X[0]<LIMIT_X[1] and LIMIT_Y[0]<LIMIT_Y[1] and LIMIT_Z[0]<LIMIT_Z[1],\
        "the limits for the workspace constraints are invalid (lower limit greater than higher limit)"

    if len(waypoints.shape) == 1:
        waypoints[0] = np.clip(waypoints[0], LIMIT_X[0], LIMIT_X[1])
        waypoints[1] = np.clip(waypoints[1], LIMIT_Y[0], LIMIT_Y[1])
        waypoints[2] = np.clip(waypoints[2], LIMIT_Z[0], LIMIT_Z[1])
    else:
        waypoints[:, 0] = np.clip(waypoints[:, 0], LIMIT_X[0], LIMIT_X[1])
        waypoints[:, 1] = np.clip(waypoints[:, 1], LIMIT_Y[0], LIMIT_Y[1])
        waypoints[:, 2] = np.clip(waypoints[:, 2], LIMIT_Z[0], LIMIT_Z[1])
    return waypoints


'''    Real-time low pass filter    '''
class real_time_filters():
    def __init__(self, cut_off_freq, sample_rate):
        """
        Parameters
        ==========
        cuttoff_freq: float
        sample_rate: float
        x0 : float
            The current unfiltered signal, x_i
        x1 : float
            The unfiltered signal at the previous sampling time, x_i-1.
        x2 : float
            The unfiltered signal at the second previous sampling time, x_i-2.
        y1 : float
            The filtered signal at the previous sampling time, y_i-1.
        y2 : float
            The filtered signal at the second previous sampling time, y_i-2.
        """
        nyquist_freq = 0.5 * sample_rate
        Wn = cut_off_freq / nyquist_freq
        self.b, self.a = butter(2, Wn, btype='lowpass')
        
        self.x = [0.] * 3     # history of previous two unfiltered samples
        self.y = [0.] * 2     # history of previous two filtered samples
        self.dt = 0
        
    def butter_lowpass(self, x0):
        a = self.a
        b = self.b
        
        # update the history of unfiltered signal samples
        self.x[2] = self.x[1]
        self.x[1] = self.x[0]
        self.x[0] = x0
        
        # start at the third sample
        if self.dt < 2:
            y = x0
        else:
            y = -a[1] * self.y[0] - a[2] * self.y[1] + \
                b[0] * self.x[0] + b[1] * self.x[1] + b[2] * self.x[2]

            # update the history of filtered signal samples
            self.y[1] = self.y[0]
            self.y[0] = y
            
        self.dt += 1
        return y
    

class Rotation(object):
    def __init__(self):
          pass
    
    def quat_conjugate(self, q):
        # quaternion as [x, y, z, w]
        negative_multiplier = np.array([-1., -1., -1., 1.])
        q_star = negative_multiplier * q
        return q_star
    
    def quat_inv(self, q):
        # quaternion as [x, y, z, w]
        q_inv = self.quat_conjugate(q) / np.linalg.norm(q)
        return q_inv
    
    def quat_neg(self, q):
        # quaternion as [x, y, z, w]
        qnew = -1. * q.copy()
        qnew[-1] = q[-1]
        return qnew

    def quat_multiply(self, q0, q1):
        # Extract the values from q0
        x0 = q0[0]
        y0 = q0[1]
        z0 = q0[2]
        w0 = q0[3]
        
        # Extract the values from q1
        x1 = q1[0]
        y1 = q1[1]
        z1 = q1[2]
        w1 = q1[3]
        
        # Computer the product of the two quaternions, term by term
        Q0Q1_w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
        Q0Q1_x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
        Q0Q1_y = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
        Q0Q1_z = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1
        
        # Create a 4 element array containing the final quaternion
        final_quaternion = np.array([Q0Q1_x, Q0Q1_y, Q0Q1_z, Q0Q1_w])
        final_quaternion /= np.linalg.norm(final_quaternion)
        
        # Return a 4 element array containing the final quaternion (q02,q12,q22,q32) 
        return final_quaternion

    def quat_diff(self, q1, q2, t=1):
        # quaternion as [x, y, z, w]
        # returns q1 - q2
        dot_prod = np.dot(q1, q2)
        
        if dot_prod < 0.:
            diff = self.quat_multiply(self.quat_multiply(q2, self.quat_inv(self.quat_neg(q1)))**t, q1)
        else:
            diff = self.quat_multiply(self.quat_multiply(q2, self.quat_inv(q1))**t, q1)
        diff /= np.linalg.norm(diff)
        
        return diff
    
    def quat2euler(self, q):
        # quaternion as [x, y, z, w]
        x, y, z, w = q
        
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)
     
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)
     
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)
     
        return [roll_x, pitch_y, yaw_z]
    
    def euler2quat(self, roll, pitch, yaw):
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        w = cy * cp * cr + sy * sp * sr
        x = cy * cp * sr - sy * sp * cr
        y = sy * cp * sr + cy * sp * cr
        z = sy * cp * cr - cy * sp * sr

        return np.array([x, y, z, w])
    

class RobotControl(object):
    def __init__(self, rosrate):
        self._MOVING_AVERAGE = 100
        self._ROSRATE = rosrate
        self.rate = 1 / self._ROSRATE.sleep_dur.to_sec()
        
        # Velocity Control
        self.joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",\
                            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        
        self.client = actionlib.SimpleActionClient(
                '/scaled_pos_joint_traj_controller/follow_joint_trajectory',
                FollowJointTrajectoryAction)
        
        self.error_wrench = np.zeros(6)
        self.error_pose = np.zeros(7)
        
        # Velocity publisher
        self.vel_pub = rospy.Publisher('/joint_group_vel_controller/command',\
                 Float64MultiArray, queue_size=100)
        
        # Error publisher
        self.pub_err_po = rospy.Publisher('/compliance_error_pose', Pose, queue_size=10)
        self.pub_err_wr = rospy.Publisher('/compliance_error_wrench', Wrench, queue_size=10)
        
        # Compliance controller publisher
        self.pub_wr = rospy.Publisher('/target_wrench', WrenchStamped, queue_size=10)
        self.pub_po = rospy.Publisher('/target_frame', PoseStamped, queue_size=10)

        # Cartesian publisher
        self.cartesian_vel_publisher = rospy.Publisher('/joint_group_vel_controller/command', Twist, queue_size=10)
        
        # Store previous joint vels for moving avg
        self.xdots = deque(maxlen=self._MOVING_AVERAGE)
        for idx in range(self._MOVING_AVERAGE):
            self.xdots.append(np.asarray([0.0] * 6))
        self.qdots = deque(maxlen=self._MOVING_AVERAGE)
        for idx in range(self._MOVING_AVERAGE):
            self.qdots.append(np.asarray([0.0] * 6))

        # Initialize filter
        cut_off_freq = 59.
        sample_rate = 120.
        self.rt_filt = real_time_filters(cut_off_freq, sample_rate)
        
        # Read current controller states
        self.sub_wr = rospy.Subscriber('/my_cartesian_compliance_controller/current_twist', TwistStamped, self.twist_cb)
        self.sub_po = rospy.Subscriber('/my_cartesian_compliance_controller/current_pose', PoseStamped, self.pose_cb)
        self.sub_force = rospy.Subscriber('/wrench', WrenchStamped, self.wrench_cb)
        self.pose_fb = None
        self.wrench_fb = None
        self.twist_fb = None

        # Service call to switch controllers
        self.switch_controller_cli = rospy.ServiceProxy('/controller_manager/switch_controller', SwitchController)
        self.list_controller_cli = rospy.ServiceProxy('/controller_manager/list_controllers', ListControllers)
        self.load_controller_cli = rospy.ServiceProxy('/controller_manager/load_controller', LoadController)
        self.unload_controller_cli = rospy.ServiceProxy('/controller_manager/unload_controller', UnloadController)

        # Hardware interface
        self.robot_states = rospy.Subscriber('/joint_states', JointState, self.joint_states_cb)
        self.joint_states = None

        self.base_link = "base_link_inertia"
        self.end_link = "tool0"
        self.robot_urdf = URDF.from_parameter_server()
        self.kdl_kin = KDLKinematics(self.robot_urdf, self.base_link, self.end_link)

        # Delay in collision checking
        self._init_check_delay = 0.5
        
        # For orienting the knife
        self.R = Rotation()
        self.zero_quaternion = self.R.euler2quat(np.deg2rad(-178.670), 
                                                 np.deg2rad(-0.155), 
                                                 np.deg2rad(-90.))
        self.knife_offset_vector = np.array([-0.095, -0.095])
        
        # Initialize joystick
        self.joystick = Joystick()
        
           
    def twist_cb(self, msg):
        try:
            if msg is not None:
                twist = [msg.twist.linear.x,
                          msg.twist.linear.y,
                          msg.twist.linear.z,
                          msg.twist.angular.x,
                          msg.twist.angular.y,
                          msg.twist.angular.z]
                self.twist_fb = twist
        except:
            pass
    
    def pose_cb(self, msg):
        try:
            if msg is not None:
                pose = [msg.pose.position.x,
                        msg.pose.position.y,
                        msg.pose.position.z,
                        msg.pose.orientation.x,
                        msg.pose.orientation.y,
                        msg.pose.orientation.z,
                        msg.pose.orientation.w]
                self.pose_fb = pose
        except Exception as e:
            print(e)
    
    def wrench_cb(self, msg):
        try:
            if msg is not None:
                wrench = [msg.wrench.force.x,
                          msg.wrench.force.y,
                          msg.wrench.force.z,
                          msg.wrench.torque.x,
                          msg.wrench.torque.y,
                          msg.wrench.torque.z]
                wrench_filt = []
                for w in wrench:
                    wrench_filt.append(self.rt_filt.butter_lowpass(w))
                self.wrench_fb = wrench_filt
        except Exception as e:
            print(e)
                
    def joint_states_cb(self, msg):
        try:
            if msg is not None:
                states = list(msg.position)
                states[2], states[0] = states[0], states[2]
                self.joint_states = tuple(states) 
        except:
            pass
        
    def robotiq_joint_state_cb(self, msg):
        try:
            if msg is not None:
                self.robotiq_joint_state = msg.position[0]
        except:
            pass
    
    def check_controller_status(self, mode):
        if mode == 'position':
            controller_name = "my_cartesian_motion_controller"
        elif mode == 'velocity':
            controller_name = "joint_group_vel_controller"
        elif mode == 'compliance':
            controller_name = "my_cartesian_compliance_controller"
        else:
            print('Not a valid controller mode')
            return 0
                
        req = ListControllersRequest()
        controller_list = self.list_controller_cli(req)
        status = 'uninitialized'
        for controller in controller_list.controller:
            if controller.name == controller_name:
                status = controller.state
        return status
    
    def unload_controller(self, mode=None):
        req = UnloadControllerRequest()
        resp = UnloadControllerResponse()
        
        if mode == "velocity":
            controller_name = "joint_group_vel_controller"
        elif mode == "position":
            controller_name = 'my_cartesian_motion_controller'
        elif mode == "compliance":
            controller_name = 'my_cartesian_compliance_controller'
        else:
            print('Not a valid controller mode')
            return 0
            
        req.name = controller_name
        
        controller_status = self.check_controller_status(mode)
        if controller_status:
            resp = self.unload_controller_cli.call(req)
            if resp.ok:
                rospy.logdebug('Unload controller {} successful'.format(mode))
                print('Unload controller {} successful'.format(mode))
        else:
            rospy.logdebug('Controller {} was already unloaded'.format(controller_name))
            print('Controller {} was already unloaded'.format(controller_name))
    
    def load_controller(self, mode=None):
        req = LoadControllerRequest()
        resp = LoadControllerResponse()
        
        if mode == "velocity":
            controller_name = "joint_group_vel_controller"
        elif mode == "position":
            controller_name = 'my_cartesian_motion_controller'
        elif mode == "compliance":
            controller_name = 'my_cartesian_compliance_controller'
        else:
            print('Not a valid controller mode')
            return 0
            
        req.name = controller_name
        
        controller_status = self.check_controller_status(mode)
        if controller_status == 'uninitialized':
            resp = self.load_controller_cli.call(req)
            if resp.ok:
                rospy.logdebug('Load controller {} successful'.format(mode))
                print('Load controller {} successful'.format(mode))
        else:
            rospy.logdebug('Controller {} was already loaded'.format(controller_name))
            print('Controller {} was already loaded'.format(controller_name))
            
    def switch_controller(self, mode):
        if mode is not None:
            status = self.check_controller_status(mode)
            if status == 'uninitialized':
                self.load_controller(mode)
            elif status == 'running':
                return 0

        req = SwitchControllerRequest()
        res = SwitchControllerResponse()

        req.start_asap = False
        req.timeout = 0.0
        
        # get list of controllers
        list_req = ListControllersRequest()
        controller_list = ListControllersResponse()
        controller_list = self.list_controller_cli(list_req)
        pos_controller_state, vel_controller_state, cartesian_compliant_state = ["stopped"] * 3
        
        for controller in controller_list.controller:
            if controller.name == "joint_group_vel_controller":
                vel_controller_state = controller.state
            elif controller.name == "my_cartesian_motion_controller":
                pos_controller_state = controller.state
            elif controller.name == "my_cartesian_compliance_controller":
                cartesian_compliant_state = controller.state

        req.stop_controllers = []
        req.start_controllers = []
        if mode == 'velocity':
            if not pos_controller_state in ["stopped", "initialized"]:
                req.stop_controllers.append('my_cartesian_motion_controller')
            if not vel_controller_state == "running":
                req.start_controllers = ['joint_group_vel_controller']
            if not cartesian_compliant_state in ["stopped", "initialized"]:
                req.stop_controllers.append('my_cartesian_compliance_controller')
            req.strictness = req.STRICT
        elif mode == 'position':
            if not pos_controller_state == "running":
                req.start_controllers = ['my_cartesian_motion_controller']
            if not vel_controller_state in ["stopped", "initialized"]:
                req.stop_controllers.append('joint_group_vel_controller')
            if not cartesian_compliant_state in ["stopped", "initialized"]:
                req.stop_controllers.append('my_cartesian_compliance_controller')
            req.strictness = req.STRICT
        elif mode =='compliance':
            if not pos_controller_state in ["stopped", "initialized"]:
                req.stop_controllers.append('my_cartesian_motion_controller')
            if not vel_controller_state in ["stopped", "initialized"]:
                req.stop_controllers.append('joint_group_vel_controller')
            if not cartesian_compliant_state == "running":
                req.start_controllers = ['my_cartesian_compliance_controller']
            req.strictness = req.STRICT
        else:
            # stop all controllers
            if pos_controller_state == 'running':
                req.stop_controllers.append('my_cartesian_motion_controller')
            if vel_controller_state == 'running':
                req.stop_controllers.append('joint_group_vel_controller')
            if cartesian_compliant_state == 'running':
                req.stop_controllers.append('my_cartesian_compliance_controller')

        res = self.switch_controller_cli.call(req)
        if res.ok:
            rospy.logdebug('Switch controller to {} successful'.format(mode))
    
    def read_pose(self):
        while self.pose_fb is None:
            continue
        return np.array(self.pose_fb)
    
    def joint2pose(self, q=None):
        if q is None:
            q = self.joint_states
        state = self.kdl_kin.forward(q)
        pos = np.array(state[:3,3]).T
        pos = pos.squeeze().tolist()
        R = state[:,:3][:3]
        euler = euler_from_matrix(R)
        
        return pos + list(euler)
    
    def go2goal(self, goal=None, max_time=15., mode=None, joint_space=False):
        if not joint_space:
            assert len(goal) == 7, "goal should be a 7-d pose vector (position, quaternion)"
        else:
            assert len(goal) == 6, "goal should be a 6-d vector (joint angles)"
            pose = self.joint2pose(q=goal)
            new_goal = np.zeros(7)
            new_goal[:3] = copy.copy(pose[:3])
            new_goal[3:] = quaternion_from_euler(*pose[3:])
            goal = new_goal.copy().tolist()
        
        goal = constrain_wayp(goal).tolist()
        
        if mode is None:
            self.switch_controller(mode='compliance')
        else:
            self.switch_controller(mode=mode)
                
        ticker = 0.
        dt = 0.5 / self.rate
        done = False
        new_goal = True
        
        while self.pose_fb is None:
            continue
        
        traj = [[0.] + self.pose_fb[:3] + goal[3:],
                [max_time] + goal]
        traj_fcn = Trajectory(traj)
        
        while not done:
            diff = np.linalg.norm(np.array(goal)[:3] - np.array(self.pose_fb)[:3])
            
            # read joystick
            _, end, pause, _, _  = self.joystick.getInput()
            if pause:
                print('paused')
                while True:
                    _, _, _, _, resume = self.joystick.getInput()
                    if resume:
                        break
            if end:
                self.stop()
                exit()
            
            # publish the commands
            waypoint = traj_fcn.get_waypoint(ticker)
            self.publish_pose(waypoint)
            
            if diff < 0.01 or ticker >= max_time:
                done = True
            ticker += dt
        
    def follow_traj(self, times, traj, trajname=None):
        
        
        
        # # load predefined trajectories
        # predefined_waypoints = np.array(json.load(open('./predefined_trajectories/{}'.format(trajname), 'r')))
        # predefined_waypoints[:, 2] = 0.178
        # predefined_times = np.linspace(0., times[-1], len(predefined_waypoints))
        # traj = np.column_stack((predefined_times, predefined_waypoints[:, :3]))

        
        
        traj[:, 1:] = constrain_wayp(traj[:, 1:])
        traj_fcn = Trajectory(traj)

        if times is None:
            times = [traj[-1, 0]]
        start_t = times[0]
        end_t = times[-1]
        
        t = start_t
        dt = 0.5 / self.rate
        done = False
        
        robot_orientation = self.zero_quaternion.copy()
        # send robot to start position
        goal = np.zeros(7)
        goal[:3] = traj_fcn.get_waypoint(t)[:3]
        goal[3:] = robot_orientation.copy()
        self.go2goal(goal, max_time=10., mode='compliance')
        self.switch_controller(mode='compliance')
        
        
        
        # waypoint_idx = 0
        prev_delta_yaw = None
        
        
        
        while not done:
            _, end, pause, _, _ = self.joystick.getInput()      # z(axes), A, B, X, stop
            if end: # to exit from code
                self.stop()
                exit()

            if pause:
                print('paused')
                while True:
                    _, _, _, _, resume = self.joystick.getInput()
                    if resume:
                        break
            
            ''' Next waypoint '''
            des_s = np.zeros(7)

            # orientation
            delta_yaw = traj_fcn.get_segment_orientation(t) - np.pi / 2.
            delta_yaw_quat = self.R.euler2quat(0., 0., delta_yaw)
            new_orientation = self.R.quat_multiply(delta_yaw_quat, self.zero_quaternion)         
            des_s[3:] = new_orientation.copy()      # quaternion
            
            # position
            waypoint = np.array(traj_fcn.get_waypoint(t))
            waypoint = constrain_wayp(waypoint)
            des_s[:3] = waypoint[:3]
            
            if prev_delta_yaw is not None:
                print(abs(delta_yaw - prev_delta_yaw))
                if (abs(delta_yaw - prev_delta_yaw) >= 1e-5):
                    print("Correcting course to algin knife with the cut...")
                    self.correct_course(waypoint, delta_yaw, new_orientation)
            prev_delta_yaw = delta_yaw
            
            
            
            # if np.any(np.abs(t - traj[waypoint_idx:, 0]) <= dt):
            #     print('changing')
            #     waypoint_idx = min(waypoint_idx + 1, len(predefined_waypoints) - 1)
            # orientation = predefined_waypoints[waypoint_idx, 3:]
            # des_s[3:] = orientation.copy()


                
            target_po = self.publish_pose(des_s)
            self.publish_feedback(target_po)            
            
            if t >= end_t:
                done = True
                
            if done:
                self.stop()
                return done
            
            t += dt
            self._ROSRATE.sleep()

    def correct_course(self, waypoint, delta_yaw, new_orientation):
        # lift the knife
        current_pose = self.read_pose()
        raised_pose = current_pose.copy()
        raised_pose[2] = 0.25
        self.go2goal(raised_pose, max_time=3.)
        
        # go to the corrected pose
        correct_pose = np.zeros(7)
        correct_pose[3:] = new_orientation.copy()
        
        knife_zero_vector = self.knife_offset_vector.copy()
        delta_R = np.array([[np.cos(delta_yaw), -np.sin(delta_yaw)],
                            [np.sin(delta_yaw), np.cos(delta_yaw)]])
        knife_new_vector = np.matmul(delta_R, knife_zero_vector)
        delta_vector = knife_zero_vector - knife_new_vector
        waypoint[0] += delta_vector[0]
        waypoint[1] += delta_vector[1]
        waypoint = constrain_wayp(waypoint)
        correct_pose[:2] = waypoint[:2]
        correct_pose[2] = 0.25
        self.go2goal(correct_pose, max_time=3.)
            
        # lower the knife
        current_pose = self.read_pose()
        lowered_pose = current_pose.copy()
        lowered_pose[2] = waypoint[2]
        self.go2goal(lowered_pose, max_time=3.)
            
    def send(self, joystick_action, initial_pose, mode_rotation):
        xdot, ydot, zdot, adot, bdot, cdot = joystick_action
        curr_pose = self.read_pose()
        new_pose = initial_pose.copy()
        feedback_pose = initial_pose.copy()
        if not mode_rotation:
            if xdot != 0:
                new_pose[0] += xdot
                feedback_pose[0] = curr_pose[0]
            if ydot != 0:
                new_pose[1] += ydot
                feedback_pose[1] = curr_pose[1]
            if zdot != 0:
                new_pose[2] += zdot
                feedback_pose[2] = curr_pose[2]
            
        else:
            adot *= 5
            bdot *= 5
            cdot *= 5

            roll_quat = self.R.euler2quat(adot, 0., 0.)
            pitch_quat = self.R.euler2quat(0., bdot, 0.)
            yaw_quat = self.R.euler2quat(0., 0., cdot)
            
            orientation = self.R.quat_multiply(yaw_quat, np.array(self.pose_fb[3:]))
            new_pose[3:] = orientation.copy()

        target_pose = self.publish_pose(new_pose)
        self.publish_feedback(target_pose)
        return feedback_pose
    
    def publish_pose(self, des_pose=[0.] * 7):
        assert len(des_pose)==7, "desired pose must be 7 dimesnional (position, quaternion). But got pose of size {}".format(len(des_pose))
        po = PoseStamped()
        po.header.frame_id = "base"
        
        des_pose = constrain_wayp(des_pose)
        
        po.pose.position.x = des_pose[0]
        po.pose.position.y = des_pose[1]
        po.pose.position.z = des_pose[2]
        
        po.pose.orientation.x = des_pose[3]
        po.pose.orientation.y = des_pose[4]
        po.pose.orientation.z = des_pose[5]
        po.pose.orientation.w = des_pose[6]
        
        self.pub_po.publish(po)
        return po
        
    def publish_wrench(self, des_wrench=[0.0] * 6):
        assert len(des_wrench)==6, "desired wrench must be 6 dimesnional (force, torque). But got pose of size {}".format(len(des_wrench))
        wr = WrenchStamped()
        wr.wrench.force.x = des_wrench[0]
        wr.wrench.force.y = des_wrench[1]
        wr.wrench.force.z = des_wrench[2]
        
        wr.wrench.torque.x = des_wrench[3]
        wr.wrench.torque.y = des_wrench[4]
        wr.wrench.torque.z = des_wrench[5]

        self.pub_wr.publish(wr)
        return wr
    
    def publish_feedback(self, target_po=None, target_wr=None):
        if not rospy.get_param('/my_cartesian_compliance_controller/solver/publish_state_feedback'):
            rospy.set_param('/my_cartesian_compliance_controller/solver/publish_state_feedback', True)
        err_pose_msg, err_wrench_msg = [None] * 2
        error_pose, error_wrench = [None] * 2
        if target_wr is not None and self.wrench_fb is not None:
            des_wrench = np.array([target_wr.wrench.force.x, target_wr.wrench.force.y, target_wr.wrench.force.z,
                                  target_wr.wrench.torque.x, target_wr.wrench.torque.y, target_wr.wrench.torque.z])
            curr_wrench = np.array(self.wrench_fb)
            error_wrench = des_wrench - curr_wrench
            self.error_wrench = np.copy(error_wrench)
            
            err_wrench_msg = Wrench()
            err_wrench_msg.force.x, err_wrench_msg.force.y, err_wrench_msg.force.z = error_wrench[:3]
            err_wrench_msg.torque.x, err_wrench_msg.torque.y, err_wrench_msg.torque.z = error_wrench[3:]
        
        if target_po is not None and self.pose_fb is not None:
            des_pose = np.array([target_po.pose.position.x, target_po.pose.position.y, target_po.pose.position.z, 
                                 target_po.pose.orientation.x, target_po.pose.orientation.y, target_po.pose.orientation.z, target_po.pose.orientation.w])
            curr_pose = np.array(self.pose_fb)
            error_position = des_pose[:3] - curr_pose[:3]
            error_orientation = self.R.quat_diff(curr_pose[3:], des_pose[3:])
            error_euler = self.R.quat2euler(error_orientation)
            error_pose = np.concatenate((error_position, error_euler))
            self.error_pose = np.copy(error_pose)
            
            err_pose_msg = Pose()
            err_pose_msg.position.x, err_pose_msg.position.y, err_pose_msg.position.z = error_position
            err_pose_msg.orientation.x, err_pose_msg.orientation.y, err_pose_msg.orientation.z, err_pose_msg.orientation.w = error_orientation
        
        if err_pose_msg is not None:
            self.pub_err_po.publish(err_pose_msg)
        if err_wrench_msg is not None:
            self.pub_err_wr.publish(err_wrench_msg)
        
        return error_pose, error_wrench
    
    def stop(self):
        self.switch_controller(mode='position')
        # self.unload_controller(mode='compliance')
