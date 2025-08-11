'''
Test script to test the whole pipeline on the robot. It reads frames from camera, fits a mask to detect start and goal
objects, and takes the robot through a straight line from start to goal
'''
import numpy as np
import matplotlib.pyplot as plt
from utilities.robot_utils import RobotControl
import rospy
import argparse
from utils import *
import os
import json
from mpl_toolkits.mplot3d import Axes3D
import cv2
from plan_trajectory import PlanTrajectory
from utils import transform
from utilities.gui import GUI


dirs = ['config/']
for dir in dirs:
    if not os.path.exists('./{}'.format(dir)):
        os.makedirs('./{}'.format(dir))


with open('./config/robot_home.json', 'r') as f:
    HOME = np.array(json.load(f))

'''
INITIALIZATION
'''

# Robot
print('')
print('[*]')
print('[*] Initializing the robot')

rospy.init_node('run_demo')
rate = 1000
rosrate = rospy.Rate(rate)
robot = RobotControl(rosrate)

print('')
print('[*]')
print('[*] Going home...')
robot.go2goal(goal=HOME, max_time=6.)

                
try:
    traj = np.array([[0., 0.15, -0.65, 0.17],
                     [5., -0.15, -0.65, 0.17],
                     [10., -0.15, -0.35, 0.17]])
    
    print('')
    print('Executing trajectory...')
    done = robot.follow_traj(traj[:, 0], traj)
    
    robot.go2goal(HOME)

finally:
    robot.switch_controller(mode='position')
    print('')
    print('[*]')
    print('[*] Terminating Code')



