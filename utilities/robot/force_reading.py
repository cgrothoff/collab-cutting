import rospy
import time
import sys
import numpy as np
from utilities.robot.robot_utils import RobotControl, Rotation


def main():
    rospy.init_node("force reading testing")
    rate = 1000
    rosrate = rospy.Rate(rate)
    
    robot = RobotControl(rosrate)
    R = Rotation()

if __name__ == '__main__':
    main()