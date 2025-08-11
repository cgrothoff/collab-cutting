#!/usr/bin/env python

import rospy
import time
import sys
import json
import numpy as np
from utilities.robot.robot_utils import RobotControl, Rotation
from utilities.robot.joystick import Joystick

with open('./config/robot_home.json', 'r') as f:
    HOME = np.array(json.load(f))
        
with open('./config/robot_skewed_home.json', 'r') as f:
    SKEWED_HOME = np.array(json.load(f))

# Last home position:
# [0.03227752280043474, -0.6246765549556433, 
# 0.2894989749757967, -0.6437785202975237, 
# 0.7644083818189861, -0.03480174998948464, 
# -0.0042285702169248904]

def main():
    rospy.init_node("teleop")
    rate = 1000
    rosrate = rospy.Rate(rate)

    robot = RobotControl(rosrate)
    joystick = Joystick()
    R = Rotation()

    print("[*] Initialized, Moving Home")
    robot.go2goal(goal=HOME, max_time=6.)
    
    # # time.sleep(2)
    # robot.go2goal(goal=SKEWED_HOME, max_time=6.)
    
    # robot.switch_controller(mode='compliance')
    print("[*] Ready for joystick inputs")

    save_traj = []
    mode_rotation = False
    initial_pose = robot.read_pose()
    while not rospy.is_shutdown():
        axes, a, b, x, stop = joystick.getInput()
        
        if stop:
            robot.send([0., 0., 0., 0., 0., 0.], initial_pose, False)
            print('')
            print('[*]')
            if len(save_traj) != 0:
                try:
                    savename = sys.argv[1]
                except:
                    savename = 'robot_coordinates_calibration'
                saveloc = './config/{}.json'.format(savename)
                print('[*] Saving trajectory to {}'.format(saveloc))
                with open(saveloc, 'w') as f:
                    json.dump(save_traj, f)
            print('[*] Terminating Code')
            exit()
        
        joystick.getAction(axes)
        initial_pose = robot.send(joystick.action, initial_pose, mode_rotation)
               
        if a:
            print('')
            print(robot.pose_fb)
            print('')
            save_traj.append(robot.pose_fb[:2])

            time.sleep(1)
        if b:
            print('')
            print('[*]')
            print('[*] switching to rotation control')
            mode_rotation = True
            time.sleep(1)
        if x:
            print('')
            print('[*]')
            print('[*] switching to translation control')
            mode_rotation = False
            time.sleep(1)
        
        
        rosrate.sleep()
        
        

    
if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass






















