import os
import json
import rospy

import numpy as np

from utilities.robot.robot_utils import RobotControl


def main():
    with open("./config/robot_coordinates_calibration.json", "r") as f:
        coordinates = np.stack(json.load(f))
    times = np.linspace(0, 9, len(coordinates))[:, None]
    h = np.ones((len(coordinates), 1)) * 0.2
    traj = np.hstack((times, coordinates, h))
    
    with open('./config/robot_home.json', 'r') as f:
        HOME = np.array(json.load(f))
    
    rospy.init_node('test_coordinates')
    rate = 1000
    rosrate = rospy.Rate(rate)
    robot = RobotControl(rosrate)
    
    print('')
    print('[*]')
    print('[*] Going home...')
    robot.go2goal(goal=HOME, max_time=6.)
    
    print('')
    print('Executing trajectory...')
    done = robot.follow_traj(traj[:, 0], traj)
    
    
if __name__ == '__main__':
    main()
    
    