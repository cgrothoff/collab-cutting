import cv2
import numpy as np
import json
from scipy.optimize import minimize
from utils import *
from utilities.robot.robot_utils import RobotControl
import rospy

np.set_printoptions(precision=3, suppress=False)



# Move knife 90 degrees to take photo:
rospy.init_node('run_demo')
rate = 1000
rosrate = rospy.Rate(rate)
robot = RobotControl(rosrate)

with open('./config/robot_skewed_home.json', 'r') as f:
    SKEWED_HOME = np.array(json.load(f))
robot.go2goal(goal=SKEWED_HOME, max_time=6.)

# robot coordinates
with open('./config/robot_coordinates_calibration.json', 'r') as f:
    robot_coordinates = np.array(json.load(f))
    
# get image
camera = cv2.VideoCapture(-1)
for idx in range(30):
    ret, img = camera.read()

cv2.imshow("test", img)
cv2.waitKey(1000)

# get my image thresholds
color_thresholds = np.array(json.load(open("./config/corners.json", "r")))
thresh_img = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), color_thresholds[0], color_thresholds[1])

cv2.imshow("test", thresh_img)
cv2.waitKey(1000)
cv2.destroyAllWindows()

# find countours 
all_corners, _ = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# get largest areas
# sort from largest to smallest area
areas = []
for corner in all_corners:
    a = -cv2.contourArea(corner)
    areas.append(a)
sorted_corners = [x for _, x in sorted(zip(areas, all_corners), key=lambda pair: pair[0])] 
# then choose the 4 items with largest areas
valid_corners = sorted_corners[0:len(robot_coordinates)]

# get their centroids
pixel_coords = []
for corner in valid_corners:
    bounding_box = cv2.boundingRect(corner)
    xy = [bounding_box[0] + bounding_box[2] / 2., bounding_box[1] + bounding_box[3] / 2.]
    pixel_coords.append(xy)
pixel_coords = sort_coordinates(pixel_coords)

debug_image = thresh_img.copy()
for i, p in enumerate(pixel_coords.tolist()):
    debug_image = cv2.circle(debug_image, (int(p[0]), int(p[1])), 5, 
                             (255, 0, 0), -1)
    debug_image = cv2.putText(img=debug_image, 
                              text=str(i + 1),
                              org=(int(p[0]) + 20, int(p[1]) + 20),
                              fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                              fontScale=1,
                              color=(255, 0, 0),
                              thickness=1)

cv2.imshow('centroids', debug_image)
cv2.waitKey(5000)

print('')
print('[*] Found {} pixel coordinates to match to {} robot coordinates'.format(len(pixel_coords), len(robot_coordinates)))

# Calculate the tansformation
M, inliers = cv2.estimateAffine2D(pixel_coords, robot_coordinates)
T_affine = np.zeros((3, 3))
T_affine[2, 2] = 1
T_affine[:2,:3] = M
            
transformation = {'T': T_affine.tolist(), 'px_coords': pixel_coords.tolist()}
json.dump(transformation, open("./config/transformation.json", "wb"))

# print example of the mapping
print("[*] Here is my mapping from pixels to robot coordinates:")
pixel_coords = pixel_coords.transpose()
pixel_coords_homogeneous = np.row_stack((pixel_coords, np.ones(pixel_coords.shape[1])))
p_guess = np.matmul(T_affine, pixel_coords_homogeneous)
p_guess = p_guess[:2, :]
p_guess = p_guess.transpose()

with np.printoptions(precision=4, suppress=True):
    for idx in range(len(robot_coordinates)):
        print(robot_coordinates[idx, :], '  -->  ', p_guess[idx, :])
    total_error = np.linalg.norm(robot_coordinates - p_guess, axis=1)
    print('Error\n', total_error)