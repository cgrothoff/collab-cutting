import rospy
import time
import sys
import json
import numpy as np
from utilities.robot.robot_utils import RobotControl, Rotation
from utilities.robot.joystick import Joystick


def main():
    rospy.init_node("teleop")
    rate = 1000
    rosrate = rospy.Rate(rate)

    robot = RobotControl(rosrate)
    robot.switch_controller(mode='compliance')

    t = 0
    j = robot.joint_states
    while j is None and t < 10000:
        j = robot.joint_states
    
    print('Joints\n', robot.joint_states)
    print('EE\n', robot.joint2pose())
    

if __name__ == '__main__':
    main()