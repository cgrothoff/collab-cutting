import rospy
import numpy as np
import time
from tqdm import tqdm
from pyquaternion import Quaternion
import actionlib
from utils import wrap_angle, recenter_zero, euler_from_R
import socket
from urdf_parser_py.urdf import URDF
from joystick import Joystick
from pykdl_utils.kdl_kinematics import KDLKinematics
from collections import deque
from std_msgs.msg import Float64MultiArray
from robotiq_2f_gripper_msgs.msg import (
    CommandRobotiqGripperFeedback, 
    CommandRobotiqGripperResult, 
    CommandRobotiqGripperAction, 
    CommandRobotiqGripperGoal
)
from robotiq_2f_gripper_control.robotiq_2f_gripper_driver import (
    Robotiq2FingerGripperDriver as Robotiq
)
from controller_manager_msgs.srv import (
    SwitchController, 
    SwitchControllerRequest, 
    SwitchControllerResponse
)
from control_msgs.msg import (
    FollowJointTrajectoryAction,
    FollowJointTrajectoryGoal,
    GripperCommandAction,
    GripperCommandGoal,
    GripperCommand
)
from trajectory_msgs.msg import (
    JointTrajectoryPoint
)
from sensor_msgs.msg import (
    JointState
)
from geometry_msgs.msg import(
    TwistStamped,
    Twist
)


'''
Class for controlling UR10
'''
STEP_SIZE_L = 0.15
STEP_SIZE_A = 0.2 * np.pi / 4
STEP_TIME = 0.01
DEADBAND = 0.1
MOVING_AVERAGE = 100
HOME_UR = [-1.2827160994159144, -1.2536433378802698, -2.1946690718280237, -1.2545693556415003, 1.5669794082641602, 0.2997606694698334]

class UR10(object):
    def __init__(self, HOME=HOME_UR):
        self.HOME = HOME

        # Action client for joint move commands
        self.client = actionlib.SimpleActionClient(
                '/scaled_pos_joint_traj_controller/follow_joint_trajectory',
                FollowJointTrajectoryAction)
        self.client.wait_for_server()
        # Velocity commands publisher
        self.vel_pub = rospy.Publisher('/joint_group_vel_controller/command',\
                 Float64MultiArray, queue_size=10)
        # Cartesian publisher
        self.cartesian_vel_publisher = rospy.Publisher('/twist_controller/command',\
                 Twist, queue_size=10)
        # Subscribers to update joint state
        self.joint_sub = rospy.Subscriber('/joint_states', JointState, self.joint_states_cb)
        # service call to switch controllers
        self.switch_controller_cli = rospy.ServiceProxy('/controller_manager/switch_controller',\
                 SwitchController)
        self.joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",\
                            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
        self.base_link = "base_link"
        self.end_link = "wrist_3_link"
        self.joint_states = None
        self.robot_urdf = URDF.from_parameter_server()
        self.kdl_kin = KDLKinematics(self.robot_urdf, self.base_link, self.end_link)
        
        # Gripper action and client
        action_name = rospy.get_param('~action_name', 'command_robotiq_action')
        self.robotiq_client = actionlib.SimpleActionClient(action_name, \
                                CommandRobotiqGripperAction)
        self.robotiq_client.wait_for_server()
        # Initialize gripper
        goal = CommandRobotiqGripperGoal()
        goal.emergency_release = False
        goal.stop = False
        goal.position = 1.00
        goal.speed = 0.1
        goal.force = 5.0
        # Sends the goal to the gripper.
        self.robotiq_client.send_goal(goal)

        # store previous joint vels for moving avg
        self.qdots = deque(maxlen=MOVING_AVERAGE)
        for idx in range(MOVING_AVERAGE):
            self.qdots.append(np.asarray([0.0] * 6))
        self.xdots = deque(maxlen=MOVING_AVERAGE)
        for idx in range(MOVING_AVERAGE):
            self.xdots.append(np.asarray([0.0] * 6))

    def joint_states_cb(self, msg):
        try:
            if msg is not None:
                states = list(msg.position)
                states[2], states[0] = states[0], states[2]
                self.joint_states = tuple(states)
        except:
            pass
    
    def switch_controller(self, mode=None):
        print('')
        print('Switching to {} control mode...'.format(mode))
        req = SwitchControllerRequest()
        res = SwitchControllerResponse()

        req.start_asap = False
        req.timeout = 0.0
        if mode == 'position':
            req.start_controllers = ['scaled_pos_joint_traj_controller']
            req.stop_controllers = ['twist_controller']
            req.strictness = req.STRICT
        elif mode == 'twist':
            req.start_controllers = ['twist_controller']
            req.stop_controllers = ['scaled_pos_joint_traj_controller']
            req.strictness = req.STRICT
        else:
            rospy.logwarn('Unkown mode for the controller!')

        res = self.switch_controller_cli.call(req)
        print('done')

    def xdot2qdot(self, xdot):
        J = self.kdl_kin.jacobian(self.joint_states)
        J_inv = np.linalg.pinv(J)
        return J_inv.dot(xdot)
    
    def wrap_angles(self, pose):
        assert len(pose) == 6,"Robot pose must be 6 dimensional. Received a {} with {} dimension".format(type(pose), len(pose))
        for i in range(3, len(pose)):
            theta = wrap_angle(pose[i])
            if i == 3:
                theta = recenter_zero(theta)
            pose[i] = theta
        return pose

    def pose2joint(self, pose):
        ''' Uses zyx convention '''
        theta = [pose[3], pose[4], pose[5]]
    
        R = np.mat([[np.cos(theta[1])*np.cos(theta[2]),       np.sin(theta[0])*np.sin(theta[1])*np.cos(theta[2]) - np.sin(theta[2])*np.cos(theta[0]),      np.sin(theta[1])*np.cos(theta[0])*np.cos(theta[2]) + np.sin(theta[0])*np.sin(theta[2]), pose[0]], \
             [np.sin(theta[2])*np.cos(theta[1]),       np.sin(theta[0])*np.sin(theta[1])*np.sin(theta[2]) + np.cos(theta[0])*np.cos(theta[2]),      np.sin(theta[1])*np.sin(theta[2])*np.cos(theta[0]) - np.sin(theta[0])*np.cos(theta[2]), pose[1]], \
                 [-np.sin(theta[1]),                        np.sin(theta[0])*np.cos(theta[1]),                                                           np.cos(theta[0])*np.cos(theta[1]), pose[2]],\
                     [0,0,0,1]])

        return self.kdl_kin.inverse(R, self.joint_states, maxiter=100000, eps=0.01)

    def joint2pose(self, joints=None):
        ''' conversion from R to euler angles uses ZYX convention '''
        if joints == None:
            joints = self.joint_states
        state = self.kdl_kin.forward(joints)
        xyz_lin = np.array(state[:,3][:3]).T
        xyz_lin[0, :2] *= -1
        xyz_lin = xyz_lin.tolist()
        R = state[:,:3][:3]
        roll, pitch, yaw = euler_from_R(R)
        xyz_ang = [roll, pitch, yaw]
        xyz = np.asarray(xyz_lin[-1]).tolist() + np.asarray(xyz_ang).tolist()
        return xyz
    
    def go2goal(self, goal_pose, exec_time=0.5, eps=0.001, action_scale=0.5):
        assert len(goal_pose) in [3, 6], 'The pose is not valid. Pose must be 3 or 6 dimensional, got {} dimensions instead'.format(len(goal_pose))
        start_time = time.time()
        curr_time = time.time() - start_time
        robot_pose = self.wrap_angles(self.joint2pose())
        if len(goal_pose) == 3:
            robot_ang = self.wrap_angles(self.joint2pose())[3:]
            goal_pose = goal_pose + robot_ang
        goal_pose, robot_pose = self.workspace_contraints(goal=np.asarray(goal_pose), limit_z=[0.2, 0.7]), np.asarray(robot_pose)
        dist = np.linalg.norm(robot_pose - goal_pose)

        joystick = Joystick()
        while (dist > eps and curr_time < exec_time):
            
            _, _, _, _, stop = joystick.getInput()
            if stop:
                self.switch_controller('position')
                self.send_joint(self.joint_states, 1.0)
                print()
                print('EMERGENCY STOP ENGAGED!!!')
                exit(0)
            
            xdot = action_scale * (goal_pose - robot_pose)

            # Constrain the robot to the workspace
            xdot = self.workspace_contraints(xdot=xdot, action_scale=action_scale, limit_z=[0.2, 0.7])
            xdot[3:5] *= 0.

            self.send(xdot)
            robot_pose = self.wrap_angles(self.joint2pose())
            curr_time = time.time() - start_time
            dist = np.linalg.norm(robot_pose - goal_pose)

    def send(self, xdot, limit=0.5):
        scale_translation = np.linalg.norm(xdot[:3])
        scale_rotation = np.linalg.norm(xdot[3:])
        if scale_translation > limit:
            xdot[:3] *= limit/scale_translation
        if scale_rotation > limit:
            xdot[3:] *= limit/scale_rotation
        self.xdots = np.delete(self.xdots, 0, 0)
        xdot = np.array(xdot)
        self.xdots = np.vstack((self.xdots, xdot))
        xdot_mean = np.mean(self.xdots, axis=0).tolist()
        msg = Twist()
        msg.linear.x = xdot_mean[0]
        msg.linear.y = xdot_mean[1]
        msg.linear.z = xdot_mean[2]
        msg.angular.x = xdot_mean[3]
        msg.angular.y = xdot_mean[4]
        msg.angular.z = xdot_mean[5]
        self.cartesian_vel_publisher.publish(msg)

    def go2home(self, exec_time=5):
        print('')
        print('Going Home...')
        safe_height = 0.5
        
        self.switch_controller(mode='twist')
        if self.joint2pose()[2] < 0.3:
            while abs(safe_height - self.joint2pose()[2]) > 0.01:
                xdot = np.zeros(6)
                xdot[2] = 1. * (safe_height - self.joint2pose()[2])
                self.send(xdot, limit=0.5)

        self.switch_controller('position')
        if np.linalg.norm(np.array(self.joint_states) - np.array(self.HOME)) > 0.001:
            self.send_joint(self.HOME, exec_time)
            self.client.wait_for_result()
        self.switch_controller('twist')
        print('done')

    def send_joint(self, pos_joint, time):
        waypoint = JointTrajectoryPoint()
        waypoint.positions = pos_joint
        waypoint.time_from_start = rospy.Duration(time)
        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = self.joint_names
        goal.trajectory.points.append(waypoint)
        goal.trajectory.header.stamp = rospy.Time.now()
        self.client.send_goal(goal)
        rospy.sleep(time)

    def actuate_gripper(self, pos, speed, force):
        Robotiq.goto(self.robotiq_client, pos=pos, speed=speed, force=force, block=True)
        return self.robotiq_client.get_result()
    
    def flush(self):
        self.xdots = deque(maxlen=len(self.xdots))
        for idx in range(MOVING_AVERAGE):
            self.xdots.append(np.asarray([0.0] * 6))
        self.send(np.zeros(6))
        
    def workspace_contraints(self, goal=None, xdot=None, action_scale=None, limit_x=[-0.3, 0.3], limit_y=[-1.1, -0.3], limit_z=[0.34, 0.7]):
        if not xdot is None:
            assert action_scale is not None,"action_scale cannot be NaN"

        if not goal is None:
            goal[0] = np.clip(goal[0], limit_x[0], limit_x[1])
            goal[1] = np.clip(goal[1], limit_y[0], limit_y[1])
            goal[2] = np.clip(goal[2], limit_z[0], limit_z[1])
            return goal
        elif not xdot is None:
            s_next = self.joint2pose() + 0.001 * action_scale * xdot
            if s_next[0] < limit_x[0] or s_next[0] > limit_x[1]:
                print('!! Constraint violated in X. Limit are: {}, coordinate goes to {}. Restricted robot movement'.format(limit_x, s_next[0]))
                xdot[0] = 0.0
            if s_next[1] < limit_y[0] or s_next[1] > limit_y[1]:
                print('!! Constraint violated in Y. Limit are: {}, coordinate goes to {}. Restricted robot movement'.format(limit_y, s_next[1]))
                xdot[1] = 0.0
            if s_next[2] < limit_z[0] or s_next[2] > limit_z[1]:
                print('!! Constraint violated in Z. Limit are: {}, coordinate goes to {}. Restricted robot movement'.format(limit_z, s_next[2]))
                xdot[2] = 0.0
            return xdot
        
    def stop(self):
        self.switch_controller(mode='position')
        self.send_joint(self.joint_states, 1.)
        print('Robot STOPPED!')
        self.switch_controller(mode='twist')
            