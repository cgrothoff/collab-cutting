import math
import cv2
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import json

def sort_coordinates(coordinates):
    def x_n_y(coord):
        x, y = coord
        return x, y
    assert isinstance(coordinates, list), 'Input must be a list, got {} instead'.format(type(coordinates))
    assert all(isinstance(c, list) for c in coordinates), 'Input must be a list of lists, got list of {} instead'.format(type(coordinates[0]))
    
    sorted_coords = sorted(coordinates, key=x_n_y)        
    return np.array(sorted_coords)


def wrap_angle(angle):
    if np.isnan(angle):
        angle = 0.
    if angle <= -np.pi:
        angle += 2 * np.pi
    elif angle > np.pi:
        angle -= 2 * np.pi
    return angle


def recenter_zero(angle):
    if np.isnan(angle):
        angle = 0.
    return wrap_angle(angle + np.pi)


def angle_bet_vec(vec1, vec2):
    angle = wrap_angle(np.arctan2(vec2[1], vec2[0]) - np.arctan2(vec1[0], vec1[1]))
    return angle


def unit_vec(vec):
    return vec / np.linalg.norm(vec)

def get_edge(img, b): 
    all_objects, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    biggest_object = None
    biggest_area = 0
    for item in all_objects:
        area = cv2.contourArea(item)
        if area > biggest_area:
            biggest_area = area
            biggest_object = item
    return biggest_object

def get_edges(img, b): 
    # all_objects, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # biggest_object = None
    # biggest_area = 0
    # for item in all_objects:
    #     area = cv2.contourArea(item)
    #     if area > biggest_area:
    #         biggest_area = area
    #         biggest_object = item
    # return biggest_object
    
    mask = cv2.medianBlur(img, 5)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.erode(mask, kernel, 1)
    _, mask, _, _ = cv2.floodFill(mask, None, (10, 10), 0)
    mask = cv2.dilate(mask, kernel, 20)
    
    filter_objects, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    filter_objects = list(filter_objects)

    area_threhold = 100
    contours = []
    for item in filter_objects:
        if cv2.contourArea(item) > area_threhold:
            contours.append(item)
    
    for i in range(len(contours)):
        rect = cv2.minAreaRect(contours[i])
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        contours[i] = box
    # print(len(contours))
    # for item in contours:
    #     mask = cv2.drawContours(b, [item], -1, (255,), thickness=cv2.FILLED)
    #     cv2.imshow('mask', mask)
    #     cv2.waitKey(0)
    return contours

class SQUISH_E:
    def SED(self, p, p_pred, p_succ):
        pos_a = p_pred[1:]
        t_a = p_pred[0]
        pos_b = p[1:]
        t_b = p[0]
        pos_c = p_succ[1:]
        t_c = p_succ[0]

        v = (pos_c - pos_a) / (t_c - t_a)
        pos_proj = pos_a + v * (t_b - t_a)
        return np.linalg.norm(pos_b - pos_proj)

    def adjust_priority(self, i, traj, Q, pred, succ, pi):
        if pred[i] > -1 and succ[i] > -1:
            Q[i] = pi[i] + self.SED(traj[i], traj[int(pred[i])], traj[int(succ[i])])
        return traj, Q, pred, succ, pi

    def reduce(self, traj, Q, pred, succ, pi):
        j = int(np.nanargmin(Q))

        pi[int(succ[j])] = max(Q[j], pi[int(succ[j])])
        pi[int(pred[j])] = max(Q[j], pi[int(pred[j])])
        succ[int(pred[j])] = succ[j]
        pred[int(succ[j])] = pred[j]
        traj, Q, pred, succ, pi = self.adjust_priority(int(pred[j]), traj, Q, pred, succ, pi)
        traj, Q, pred, succ, pi = self.adjust_priority(int(succ[j]), traj, Q, pred, succ, pi)
        Q[j] = np.nan
        return traj, Q, pred, succ, pi

    def squish(self, traj, lamda=0.5, mu=0):
        """
        traj: input ndarray of size nxd, where n is the number of points and d is the dimensionality
        lambda: compression ratio between output points and input points. Lies in [0, 1].
        mu: Maximum acceptable error in the system
        returns compressed_traj with size pxd, where p <= n * lamda
        """
        beta = np.ceil(len(traj) * lamda)
        n_pts = len(traj)
        Q = np.full(n_pts, np.nan)
        pi = np.ones(n_pts) * -1
        succ = np.ones(n_pts) * -1
        pred = np.ones(n_pts) * -1

        for i in range(n_pts):
            Q[i] = np.inf
            pi[i] = 0

            if i >= 1:
                succ[i - 1] = i
                pred[i] = i - 1
                traj, Q, pred, succ, pi = self.adjust_priority(
                    i - 1, traj, Q, pred, succ, pi
                )

            q_non_zero = np.sum(np.invert(np.isnan(Q)))
            if q_non_zero > beta:
                traj, Q, pred, succ, pi = self.reduce(traj, Q, pred, succ, pi)
        # print(Q)
        p = np.nanmin(Q)
        while p <= mu:
            traj, Q, pred, succ, pi = self.reduce(traj, Q, pred, succ, pi)
            p = np.nanmin(Q)
        return traj[np.invert(np.isnan(Q))]


def R_from_euler(theta1, theta2, theta3, order='xyz'):
    """
    adopted from https://programming-surgeon.com/en/euler-angle-python-en/
    input
        theta1, theta2, theta3 = rotation angles in rotation order (radians)
        oreder = rotation order of x,y,z e.g. XZY rotation -- 'xzy' 
        output 3x3 rotation matrix (numpy array)
    """
    c1 = np.cos(theta1)
    s1 = np.sin(theta1)
    c2 = np.cos(theta2)
    s2 = np.sin(theta2)
    c3 = np.cos(theta3)
    s3 = np.sin(theta3)

    if order == 'xzx':
        matrix=np.array([[c2, -c3*s2, s2*s3],
                         [c1*s2, c1*c2*c3-s1*s3, -c3*s1-c1*c2*s3],
                         [s1*s2, c1*s3+c2*c3*s1, c1*c3-c2*s1*s3]])
    elif order=='xyx':
        matrix=np.array([[c2, s2*s3, c3*s2],
                         [s1*s2, c1*c3-c2*s1*s3, -c1*s3-c2*c3*s1],
                         [-c1*s2, c3*s1+c1*c2*s3, c1*c2*c3-s1*s3]])
    elif order=='yxy':
        matrix=np.array([[c1*c3-c2*s1*s3, s1*s2, c1*s3+c2*c3*s1],
                         [s2*s3, c2, -c3*s2],
                         [-c3*s1-c1*c2*s3, c1*s2, c1*c2*c3-s1*s3]])
    elif order=='yzy':
        matrix=np.array([[c1*c2*c3-s1*s3, -c1*s2, c3*s1+c1*c2*s3],
                         [c3*s2, c2, s2*s3],
                         [-c1*s3-c2*c3*s1, s1*s2, c1*c3-c2*s1*s3]])
    elif order=='zyz':
        matrix=np.array([[c1*c2*c3-s1*s3, -c3*s1-c1*c2*s3, c1*s2],
                         [c1*s3+c2*c3*s1, c1*c3-c2*s1*s3, s1*s2],
                         [-c3*s2, s2*s3, c2]])
    elif order=='zxz':
        matrix=np.array([[c1*c3-c2*s1*s3, -c1*s3-c2*c3*s1, s1*s2],
                         [c3*s1+c1*c2*s3, c1*c2*c3-s1*s3, -c1*s2],
                         [s2*s3, c3*s2, c2]])
    elif order=='xyz':
        matrix=np.array([[c2*c3, -c2*s3, s2],
                         [c1*s3+c3*s1*s2, c1*c3-s1*s2*s3, -c2*s1],
                         [s1*s3-c1*c3*s2, c3*s1+c1*s2*s3, c1*c2]])
    elif order=='xzy':
        matrix=np.array([[c2*c3, -s2, c2*s3],
                         [s1*s3+c1*c3*s2, c1*c2, c1*s2*s3-c3*s1],
                         [c3*s1*s2-c1*s3, c2*s1, c1*c3+s1*s2*s3]])
    elif order=='yxz':
        matrix=np.array([[c1*c3+s1*s2*s3, c3*s1*s2-c1*s3, c2*s1],
                         [c2*s3, c2*c3, -s2],
                         [c1*s2*s3-c3*s1, c1*c3*s2+s1*s3, c1*c2]])
    elif order=='yzx':
        matrix=np.array([[c1*c2, s1*s3-c1*c3*s2, c3*s1+c1*s2*s3],
                         [s2, c2*c3, -c2*s3],
                         [-c2*s1, c1*s3+c3*s1*s2, c1*c3-s1*s2*s3]])
    elif order=='zyx':
        matrix=np.array([[c1*c2, c1*s2*s3-c3*s1, s1*s3+c1*c3*s2],
                         [c2*s1, c1*c3+s1*s2*s3, c3*s1*s2-c1*s3],
                         [-s2, c2*s3, c2*c3]])
    elif order=='zxy':
        matrix=np.array([[c1*c3-s1*s2*s3, -c2*s1, c1*s3+c3*s1*s2],
                         [c3*s1+c1*s2*s3, c1*c2, s1*s3-c1*c3*s2],
                         [-c2*s3, s2, c2*c3]])

    return matrix


def euler_from_R(R):
    ''' follows ZYX convention '''
    pitch = -np.arcsin(R[2,0])
    roll = np.arctan2(R[2,1]/np.cos(pitch),R[2,2]/np.cos(pitch))
    yaw = np.arctan2(R[1,0]/np.cos(pitch),R[0,0]/np.cos(pitch))
    return roll, pitch, yaw
    

    
def transform(waypoints_px, height):
    waypoints_px = waypoints_px.transpose()
    waypoints_px = np.row_stack((waypoints_px, np.ones(waypoints_px.shape[1])))
    
    with open('./config/transformation.json', 'r') as f:
        data = json.load(f)        
    T = np.array(data['T'])
    
    heights = np.ones(waypoints_px.shape[1]) * height
    
    waypoints = np.matmul(T, waypoints_px).transpose()[:, :2]
    waypoints = np.column_stack((waypoints, heights))
    
    return waypoints
    

