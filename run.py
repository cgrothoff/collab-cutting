'''
Test script to test the whole pipeline on the robot. It reads frames from camera, fits a mask to detect start and goal
objects, and takes the robot through a straight line from start to goal
'''
import numpy as np
import matplotlib.pyplot as plt
from utilities.robot.robot_utils import RobotControl
import rospy
import argparse
from utils import *
import os
import json
from mpl_toolkits.mplot3d import Axes3D
import cv2
from plan_trajectory import PlanTrajectory
from utils import transform
from utilities.gui.gui import GUI
import time

import socket
import select
import sys

# Setting up Socket Server:
HOST = '127.0.0.1'
PORT_HAND = 38088
PORT_FORCE = 38090

#print(f'\n[*] Starting socket safety server on {HOST}:{PORT}')
safety_socket_hand = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
safety_socket_hand.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) 
safety_socket_hand.bind((HOST, PORT_HAND)) 
safety_socket_hand.listen(1) 

safety_socket_force = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
safety_socket_force.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) 
safety_socket_force.bind((HOST, PORT_FORCE)) 
safety_socket_force.listen(1) 

#safety_socket.setblocking(0) 

print("[] Waiting for safety client connection...") 
client_conn_hand, addr_hand = safety_socket_hand.accept() 
print("[] Safety client (hand) connected from ", addr_hand) 
client_conn_force, addr_force = safety_socket_force.accept() 
print("[] Safety client (force) connected from ", addr_force) 

dirs = ['config/']
for dir in dirs:
    if not os.path.exists('./{}'.format(dir)):
        os.makedirs('./{}'.format(dir))


with open('./config/robot_home.json', 'r') as f:
    HOME = np.array(json.load(f))

with open('./config/robot_skewed_home.json', 'r') as f:
    SKEWED_HOME = np.array(json.load(f))

'''
INITIALIZATION
'''
parser = argparse.ArgumentParser()
parser.add_argument('--resolution', default=80, type=int,
                    help="resolution of the contours. Controls the refinement of the curve (default: 400)")
parser.add_argument('--delta', type=int, default=10,
                    help="maximum distance allowed between two points in the trim (default: 10)")
parser.add_argument('--eps', type=float, default=10.,
                    help="distance threshold when comparing points between the two contours (default: 5.0)")
parser.add_argument('--height', type=float, default=0.12)
parser.add_argument('--traj-time', type=float, default=20.)
parser.add_argument('--traj-type', type=str, default='cut')
parser.add_argument('--transparent', action='store_true')
parser.add_argument('--feedback', action='store_true')
parser.add_argument('--manual', action='store_true')
parser.add_argument('--trim', action='store_true')
parser.add_argument('--slice', action='store_true')
parser.add_argument('--load-image', type=str, default=None)
args = parser.parse_args()

# Robot
print('')
print('[*]')
print('[*] Initializing the robot')

# rospy.init_node('run_demo')
# rate = 1000
# rosrate = rospy.Rate(rate)
# robot = RobotControl(rosrate)

print('')
print('[*]')
print('[*] Going home...')
# robot.go2goal(goal=HOME, max_time=6.)
# robot.go2goal(goal=SKEWED_HOME, max_time=6.)

# Move knife 90 degrees to take photo:

                
try:
    if args.load_image is not None:
        img = cv2.imread(args.load_image)
    else:
        camera = cv2.VideoCapture(2)
        for idx in range(100):
            ret, img = camera.read()

    cv2.imshow("Captured image", img)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
        
    kwargs = vars(args)
    #print(kwargs)
    motion_planner = PlanTrajectory(**kwargs)
    traj = motion_planner.plan(img, args)
    
    traj_name = None
    if args.manual:
        traj_name = 'manual.json'
    elif args.transparent:
        traj_name = 'transparent.json'
    elif args.feedback:
        traj_name = 'feedback.json'
    else:
        traj_name = 'auto.json'
    
    print('')
    print('[*]')
    print('[*] Executing trajectory...')
            
    # try:
    # done = robot.follow_traj(traj[:, 0], traj, client_conn_hand, client_conn_force, traj_name)
    
    # if not done:
    #     print('[!] Safety Interrupt Triggered')
    #     sys.exit(0)    

    print('')
    print('[*]')
    print('[*] Trajectory Executed')
    
    print('')
    print('[*]')
    print('[*] Going home...')
    
    print('')
    print('[*]')
    print('[*] Performing Uncertainty Test')
    
    
    #----------Uncertainty Addition----------
    # Move robot back to home position
    # robot.go2goal(HOME)
    # robot.go2goal(SKEWED_HOME)
    
    # Take photo of cut meat
    if args.load_image is not None:
        post_img = cv2.imread(args.load_image)
    else:
        for idx in range(100):
            ret, post_img = camera.read()

        cv2.imshow("test", post_img)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()
    
    # Detect uncertainty
    uncertainty = motion_planner.detect_uncertainty(post_img)
    print('[*] Uncertainty of Cut: ',uncertainty)
    print('[*] Cut Complete')   
except Exception as e:
    print('[!] Safety Interrupt Triggered: ', e)
    sys.exit(0)
        
finally:
    # robot.switch_controller(mode='position')
    
    print('')
    print('[*]')
    print('[*] Terminating Code')
    
    



