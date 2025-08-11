from utilities.robot.robot_utils import RobotControl
import rospy
import numpy as np
import json


with open('./config/robot_home.json', 'r') as f:
    HOME = np.array(json.load(f))
    
with open('./config/robot_skewed_home.json', 'r') as f:
    SKEWED_HOME = np.array(json.load(f))

rospy.init_node('run_demo')
rosrate = rospy.Rate(250)
robot = RobotControl(rosrate)

robot.go2goal(SKEWED_HOME)