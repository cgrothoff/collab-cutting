import numpy as np
import cv2
import os
import argparse
import glob
import json
from tqdm import tqdm
from itertools import permutations
from utilities.robot.trajectory import Trajectory
from utils import SQUISH_E, transform, get_edges, get_edge
from utilities.gui.gui import GUI
import math

class PlanTrajectory:
    def __init__(self, height=0.16, traj_time=20., **kwargs):
        for filename in ['fat', 'meat']:
            with open('./config/{}.json'.format(filename), 'r') as f:
                data = json.load(f)
                
                exec('self.' + filename + ' = data')
        self.traj_time = traj_time
        self.height = height
        self.pre_coords = 0
        
        for key in kwargs.keys():
            exec('self.{} = {}["{}"]'.format(key, kwargs, key))

    def plan(self, image, args):
        self.image = image
        
        if self.manual:
            self.trajectory = self.manual_planning()
        else:
            self.trajectory = self.autonomous_planning(args)
        
        return self.trajectory
    
    def show_interface(self, waypoints_px):
        if not (self.transparent or self.manual or self.feedback):
            return waypoints_px
        
        if self.transparent:
            interface_type = 'transparent'
        elif self.feedback:
            interface_type = 'feedback'
        elif self.manual:
            interface_type = 'manual'
        
        image_render = self.image
        image_render = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        gui = GUI(image_render.swapaxes(0, 1), waypoints_px, interface_type)
        gui.render()
        waypoints_px = gui.get_trajectory()
        return waypoints_px

    def autonomous_planning(self, args):
        updated_cut = self.plan_cut(args)

        # get the final trajectory
        waypoints = transform(updated_cut, self.height)
        
        traj_times = np.linspace(0., self.traj_time, waypoints.shape[0])
        trajectory = np.column_stack((traj_times, waypoints))
        return trajectory

    def get_bounding_box(self, piece_contours):
        # find contours of entire piece of meat
        #all_contours = np.vstack(piece_contours)
        #print(piece_contours)
        
        # Put bounding box around piece and get corner coordinates
        x,y,w,h = cv2.boundingRect(piece_contours)
        
        # TL , TR, BL, BR
        box_coords = np.array([
            [x, y],          # Top Left
            [x, (y + h)],      # Bottom Left
            [(x + w), y],      # Top Right
            [(x + w), (y + h)]  # Bottom Right
        ])
        
        return box_coords
        
    def manual_planning(self):
        cut = self.show_interface(None)
        
        piece_thresholds = np.array(json.load(open("./config/piece.json", "r"))) # original: ./config/piece.json
        piece_img = cv2.inRange(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB), piece_thresholds[0], piece_thresholds[1])
        cv2.imwrite('./pre.png', piece_img)
        piece_contours = get_edges(piece_img ,self.image.copy())
        
        piece_masks = []
        all_contours = []            
        for i in range(len(piece_contours)):
            piece_contours[i] = np.concatenate((piece_contours[i], [piece_contours[i][0]]), axis=0)
            piece_contours[i] = piece_contours[i].reshape((-1, 2))
            all_contours.append(piece_contours[i])
            
            meat_mask = np.zeros(self.image.shape, dtype=np.uint8)
            meat_mask = cv2.drawContours(meat_mask, [piece_contours[i]], -1, [0, 0, 255], thickness=cv2.FILLED)
            piece_masks.append(meat_mask)        
        
        piece_contour = np.vstack(all_contours)
        self.pre_coords = self.get_bounding_box(piece_contour)
        
        # get the final trajectory
        waypoints = transform(cut, self.height)
        
        traj_times = np.linspace(0., self.traj_time, waypoints.shape[0])
        trajectory = np.column_stack((traj_times, waypoints))
        
        # piece_thresholds = np.array(json.load(open("./config/piece-dough.json", "r"))) # original: ./config/piece.json
        # piece_img = cv2.inRange(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB), piece_thresholds[0], piece_thresholds[1])
        # cv2.imwrite('./pre.png', self.image)
        # piece_contours = get_edges(piece_img ,self.image.copy())
        # piece_masks = []
        # all_contours = []
        
        # for i in range(len(piece_contours)):
        #     piece_contours[i] = np.concatenate((piece_contours[i], [piece_contours[i][0]]), axis=0)
        #     piece_contours[i] = piece_contours[i].reshape((-1, 2))
        #     all_contours.append(piece_contours[i])
            
        #     meat_mask = np.zeros(self.image.shape, dtype=np.uint8)
        #     meat_mask = cv2.drawContours(meat_mask, [piece_contours[i]], -1, [0, 0, 255], thickness=cv2.FILLED)
        #     piece_masks.append(meat_mask)        
        
        # piece_contour = np.vstack(all_contours)
        # self.pre_coords = self.get_bounding_box(piece_contour)
        # print("Pre Cut Bounding Box Coordinates: ", self.pre_coords)
        
        return trajectory
        
    def plan_cut(self, args):
        if args.trim:
            # find the meat contours
            meat_thresholds = np.array(json.load(open("./config/meat.json", "r"))) #./config/meat.json"
            meat_img = cv2.inRange(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB), meat_thresholds[0], meat_thresholds[1])
            meat_contours = get_edge(meat_img, self.image.copy())
            
            meat_contours = np.concatenate((meat_contours, [meat_contours[0]]), axis=0)# for pork chops

            # meat_masks = []
            # for i in range(len(meat_contours)):
            #     meat_contours[i] = np.concatenate((meat_contours[i], [meat_contours[i][0]]), axis=0)
            #     meat_contours[i] = meat_contours[i].reshape((-1, 2))
            #     meat_mask = np.zeros(self.image.shape, dtype=np.uint8)
            #     meat_mask = cv2.drawContours(meat_mask, [meat_contours[i]], -1, [0, 0, 255], thickness=cv2.FILLED)
            #     meat_masks.append(meat_mask)
            
            # find the fat contours
            fat_thresholds = np.array(json.load(open("./config/fat.json", "r"))) #./config/fat.json"
            fat_img = cv2.inRange(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB), fat_thresholds[0], fat_thresholds[1])
            fat_contours = get_edge(fat_img, self.image.copy())
            
            fat_contours = np.concatenate((fat_contours, [fat_contours[0]]), axis=0)
            
            # fat_masks = []
            # for i in range(len(fat_contours)):
            #     fat_contours[i] = np.concatenate((fat_contours[i], [fat_contours[i][0]]), axis=0)
            #     fat_contours[i] = fat_contours[i].reshape((-1, 2))
            #     fat_mask = np.zeros(self.image.shape, dtype=np.uint8)
            #     fat_mask = cv2.drawContours(fat_mask, [fat_contours[i]], -1, [0, 0, 255], thickness=cv2.FILLED)
            #     fat_masks.append(fat_mask)
            # fat_contours = sorted(fat_contours, key=lambda x: -np.mean(x[:, 0]))

            piece_thresholds = np.array(json.load(open("./config/piece.json", "r"))) # original: ./config/piece.json
            piece_img = cv2.inRange(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB), piece_thresholds[0], piece_thresholds[1])
            cv2.imwrite('./pre.png', piece_img)
            piece_contours = get_edges(piece_img ,self.image.copy())
            
            piece_masks = []
            all_contours = []            
            for i in range(len(piece_contours)):
                piece_contours[i] = np.concatenate((piece_contours[i], [piece_contours[i][0]]), axis=0)
                piece_contours[i] = piece_contours[i].reshape((-1, 2))
                all_contours.append(piece_contours[i])
                
                meat_mask = np.zeros(self.image.shape, dtype=np.uint8)
                meat_mask = cv2.drawContours(meat_mask, [piece_contours[i]], -1, [0, 0, 255], thickness=cv2.FILLED)
                piece_masks.append(meat_mask)        
            
            piece_contour = np.vstack(all_contours)
            self.pre_coords = self.get_bounding_box(piece_contour)
            
            # self.pre_coords = self.get_bounding_box(piece_contour)
            
            
            # piece_thresholds = np.array(json.load(open("./config/dough-piece.json", "r")))
            # piece_img = cv2.inRange(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB), piece_thresholds[0], piece_thresholds[1])
            # piece_contours = get_edges(piece_img ,self.image.copy())
            # self.pre_coords = self.get_bounding_box(piece_contours)
            
            #print("Pre Cut Bounding Box Coordinates: ", self.pre_coords)
        
            
            # debug
            debug_img = self.image.copy()
            debug_img = cv2.drawContours(debug_img, [meat_contours], -1, [0, 0, 255], 5)
            debug_img = cv2.drawContours(debug_img, [fat_contours], -1, [255, 0, 0], 5)
            cv2.imshow('Identifying meat and fat', debug_img)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()


            
            ''' ================================================================== '''
            times = np.linspace(0, 1, meat_contours.shape[0])
            meat_traj = Trajectory(np.column_stack((times, meat_contours.reshape(-1, 2))))
            times = np.linspace(0, 1, fat_contours.shape[0])
            fat_traj = Trajectory(np.column_stack((times, fat_contours.reshape(-1, 2))))
            
            # find points of interest
            trim = []
            n_resolution = args.resolution
            for t_fat in tqdm(np.linspace(0, 1, n_resolution)):
                f = fat_traj.get_waypoint(t_fat)
                for t_meat in np.linspace(0, 1, n_resolution):
                    m = meat_traj.get_waypoint(t_meat)
                    valid = True
                    for item in trim:
                        dist = np.linalg.norm(m - np.array(item))
                        if dist < args.delta:
                            valid = False
                    if valid:
                        dist = np.linalg.norm(m - f)
                        if dist < args.eps:
                            trim.append(list(m))

            # find the shortest path
            possible_trims = permutations(trim)
            shortest_path = None
            min_distance = np.inf
            for item in tqdm(possible_trims):
                path = np.array(item)
                length = 0
                for idx in range(1, len(path)):
                    length += np.linalg.norm(path[idx, :] - path[idx-1,:])
                if length < min_distance:
                    min_distance = length
                    shortest_path = np.copy(path)
            trim = shortest_path
            
            SQ = SQUISH_E()
            curve_traj = np.column_stack((np.linspace(0, 1, len(trim)), trim))
            curve_traj = SQ.squish(curve_traj, lamda=0.7, mu=0.)[:, 1:]
            curve_traj_int = curve_traj.astype(int)
            
            # extend the curve on both sides
            trim_curve = np.copy(curve_traj)
            slope1 = -(curve_traj[0,:] - curve_traj[1,:])
            slope2 = curve_traj[-2,:] - curve_traj[-1,:]
            extend1a = curve_traj[0,:] - slope1 / np.linalg.norm(slope1) * 80
            extend1b = curve_traj[0,:] + slope1 / np.linalg.norm(slope1) * 80
            extend2a = curve_traj[-1,:] - slope2 / np.linalg.norm(slope2) * 80
            extend2b = curve_traj[-1,:] - slope2 / np.linalg.norm(slope2) * 80
            if np.linalg.norm(extend1a - curve_traj[1,:]) > np.linalg.norm(extend1b - curve_traj[1,:]):
                trim_curve[0,:] = extend1a
            else:
                trim_curve[0,:] = extend1b
            if np.linalg.norm(extend2a - curve_traj[-2,:]) > np.linalg.norm(extend2b - curve_traj[-2,:]):
                trim_curve[-1,:] = extend2a
            else:
                trim_curve[-1,:] = extend2b
            
            # resample the trajectory so that waypoints are evenly spaced
            times = np.linspace(0, 1, trim_curve.shape[0])
            cut_curve_traj = np.column_stack((times, trim_curve))
            traj = Trajectory(cut_curve_traj)
            resampled_curve = [traj.get_waypoint(0)]
            for t in tqdm(np.linspace(0, 1.0, 1001)):
                xy = traj.get_waypoint(t)
                if np.linalg.norm(xy - resampled_curve[-1]) > 10:
                    resampled_curve.append(xy)
            resampled_curve = np.array(resampled_curve)

            # print the close points for debugging
            curve_pixel_int = resampled_curve.astype(int)
            img = self.image.copy()
            debug_img = cv2.drawContours(img, [curve_pixel_int.reshape((-1, 1, 2))], -1, [0, 153, 255], 10)
            cv2.imshow("test", debug_img)
            cv2.waitKey(1000)

            # ensure that the cut starts from the top
            sorted_idx = np.argsort(curve_pixel_int[:, 1]).tolist()
            curve_pixel_int = curve_pixel_int[sorted_idx, :]
            
            # call the gui        
            updated_cut = resampled_curve.copy()
            updated_cut = self.show_interface(curve_pixel_int)
            
            
            ''' ================================================================== '''
            
            
            #### everything from here is for multiple meat and fat cuts on single piece
            # find points of interest
            # current_contour = fat_contours[0]
            # cut = current_contour[1:3, :]
            # cut = []
            # n_resolution = self.resolution
            # for t_fat in tqdm(np.linspace(0, 1, n_resolution)):
            #     f = fat_traj.get_waypoint(t_fat)
            #     for t_meat in np.linspace(0, 1, n_resolution):
            #         m = meat_traj.get_waypoint(t_meat)
            #         valid = True
            #         for item in cut:
            #             dist = np.linalg.norm(m - np.array(item))
            #             if dist < self.delta:
            #                 valid = False
            #         if valid:
            #             dist = np.linalg.norm(m - f)
            #             if dist < self.eps:
            #                 cut.append(list(m))

            # # find the shortest path
            # possible_cuts = permutations(cut)
            # shortest_path = None
            # min_distance = np.inf
            # num_iter = 0
            # for item in tqdm(possible_cuts):
            #     path = np.array(item)
            #     length = 0
            #     for idx in range(1, len(path)):
            #         length += np.linalg.norm(path[idx, :] - path[idx-1,:])
            #     if length < min_distance:
            #         min_distance = length
            #         shortest_path = np.copy(path)
            #     num_iter += 1
            #     if num_iter >= 1e5:
            #         break
            # cut = shortest_path

            # # Squish the close points to fit line segments
            # SQ = SQUISH_E()
            # curve_traj = np.column_stack((np.linspace(0, 1, len(cut)), cut))
            # curve_traj = SQ.squish(curve_traj, lamda=(5/len(cut)), mu=0.)[:, 1:]
            # curve_traj_int = curve_traj.astype(int)
            # debug_img = cv2.drawContours(self.image, [curve_traj_int.reshape((-1, 1, 2))], -1, [0, 0, 0], 5)
            # cv2.imshow("test", debug_img)
            # cv2.waitKey(1000)
            # cv2.destroyAllWindows()

            # # extend the curve on both sides
            # cut_curve = np.copy(curve_traj)
            # slope1 = -(curve_traj[0,:] - curve_traj[1,:])
            # slope2 = curve_traj[-2,:] - curve_traj[-1,:]
            # extend1a = curve_traj[0,:] - slope1 / np.linalg.norm(slope1) * 80
            # extend1b = curve_traj[0,:] + slope1 / np.linalg.norm(slope1) * 80
            # extend2a = curve_traj[-1,:] - slope2 / np.linalg.norm(slope2) * 80
            # extend2b = curve_traj[-1,:] - slope2 / np.linalg.norm(slope2) * 80
            # if np.linalg.norm(extend1a - curve_traj[1,:]) > np.linalg.norm(extend1b - curve_traj[1,:]):
            #     cut_curve[0,:] = extend1a
            # else:
            #     cut_curve[0,:] = extend1b
            # if np.linalg.norm(extend2a - curve_traj[-2,:]) > np.linalg.norm(extend2b - curve_traj[-2,:]):
            #     cut_curve[-1,:] = extend2a
            # else:
            #     cut_curve[-1,:] = extend2b

            # # resample the trajectory so that waypoints are evenly spaced 
            # times = np.linspace(0, 1, cut_curve.shape[0])
            # cut_curve_traj = np.column_stack((times, cut_curve))
            # traj = Trajectory(cut_curve_traj)
            # resampled_curve = [traj.get_waypoint(0)]
            # for t in tqdm(np.linspace(0, 1.0, 1001)):
            #     xy = traj.get_waypoint(t)
            #     if np.linalg.norm(xy - resampled_curve[-1]) > 10:
            #         resampled_curve.append(xy)
            # resampled_curve = np.array(resampled_curve)

            # # print the close points for debugging
            # curve_pixel_int = resampled_curve.astype(int)
            # debug_img = cv2.drawContours(self.image.copy(), [curve_pixel_int.reshape((-1, 1, 2))], -1, [0, 153, 255], 5)
            # cv2.imshow("test", debug_img)
            # cv2.waitKey(1000)
            # cv2.destroyAllWindows()

            # # ensure that the cut starts from the top
            # sorted_idx = np.argsort(curve_pixel_int[:, 1]).tolist()
            # curve_pixel_int = curve_pixel_int[sorted_idx, :]
            
            # # call the gui        
            # updated_cut = resampled_curve.copy()
            # updated_cut = self.show_interface(curve_pixel_int)
        
        elif args.slice:
            # find the piece contours
            meat_thresholds = np.array(json.load(open("./config/pork_loin.json", "r")))
            meat_img = cv2.inRange(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB), meat_thresholds[0], meat_thresholds[1])
            meat_contours = get_edges(meat_img, self.image.copy())
            meat_masks = []
            for i in range(len(meat_contours)):
                meat_contours[i] = np.concatenate((meat_contours[i], [meat_contours[i][0]]), axis=0)
                meat_contours[i] = meat_contours[i].reshape((-1, 2))
                meat_mask = np.zeros(self.image.shape, dtype=np.uint8)
                meat_mask = cv2.drawContours(meat_mask, [meat_contours[i]], -1, [0, 0, 255], thickness=cv2.FILLED)
                meat_masks.append(meat_mask)
            
            piece_thresholds = np.array(json.load(open("./config/piece.json", "r"))) # original: ./config/piece.json
            piece_img = cv2.inRange(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB), piece_thresholds[0], piece_thresholds[1])
            cv2.imwrite('./pre.png', piece_img)
            piece_contours = get_edges(piece_img ,self.image.copy())
            piece_masks = []
            all_contours = []            
            for i in range(len(piece_contours)):
                piece_contours[i] = np.concatenate((piece_contours[i], [piece_contours[i][0]]), axis=0)
                piece_contours[i] = piece_contours[i].reshape((-1, 2))
                all_contours.append(piece_contours[i])
                
                meat_mask = np.zeros(self.image.shape, dtype=np.uint8)
                meat_mask = cv2.drawContours(meat_mask, [piece_contours[i]], -1, [0, 0, 255], thickness=cv2.FILLED)
                piece_masks.append(meat_mask)        
            
            piece_contour = np.vstack(all_contours)
            self.pre_coords = self.get_bounding_box(piece_contour)
            
            # debug
            debug_img = self.image.copy()
            debug_img = cv2.drawContours(debug_img, meat_contours, -1, [0, 0, 255], 5)
            cv2.imshow('DEBUG', debug_img)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()

            # find points of interest
            current_contour = meat_contours[0]
            x, y, w, h = cv2.boundingRect(current_contour)
            
            # debug
            img = self.image.copy()
            cv2.rectangle(img, (x, y), (x + w, y + h), (255,0,0), 4)
            cv2.imshow('test', img)
            cv2.waitKey(1000)
            
            # plan slicing path           
            n_cuts = 4
            n_points = 3
            square = np.zeros(((n_cuts - 1) * n_points, 2))
            x_coords = np.linspace(0, 1, n_cuts + 1)[1:-1]
            for idx, xc in enumerate(x_coords):
                square[idx*n_points:(idx+1)*n_points, :] = np.column_stack((np.ones(n_points)*xc, np.linspace(0, 1, n_points)))
            square = square[-n_points:, :]
            
            # scale to the bounding box
            square[:, 0] = w * square[:,0] + x
            square[:, 1] = 1.4 * h * square[:,1] + y - .25*h
            slice = square

            # print slice for debugging
            slice_int = slice.astype(int)
            debug_img = cv2.drawContours(img, [slice_int.reshape((-1, 1, 2))], -1, [0, 0, 255], 2)
            cv2.imshow("test", debug_img)
            cv2.waitKey(1000)
            
            # call the gui        
            updated_cut = self.show_interface(slice_int)
        return updated_cut
    
    def detect_uncertainty(self, post_img):
        # Apply mask, generate contours, and get coordinates of bounding box post cut
        piece_thresholds = np.array(json.load(open("./config/piece.json", "r"))) # original: ./config/piece.json
        piece_img = cv2.inRange(cv2.cvtColor(post_img, cv2.COLOR_BGR2RGB), piece_thresholds[0], piece_thresholds[1])
        cv2.imwrite('./post.png', piece_img)
        piece_contours = get_edges(piece_img ,self.image.copy())
        piece_masks = []
        all_contours = []
        
        for i in range(len(piece_contours)):
            piece_contours[i] = np.concatenate((piece_contours[i], [piece_contours[i][0]]), axis=0)
            piece_contours[i] = piece_contours[i].reshape((-1, 2))
            all_contours.append(piece_contours[i])
            
            meat_mask = np.zeros(self.image.shape, dtype=np.uint8)
            meat_mask = cv2.drawContours(meat_mask, [piece_contours[i]], -1, [0, 0, 255], thickness=cv2.FILLED)
            piece_masks.append(meat_mask)        
        
        # Put bounding box around piece and get corner coordinates
        piece_contour = np.vstack(all_contours)
        post_coords = self.get_bounding_box(piece_contour)
        
        dx_total = 0
        dy_total = 0
        
        for point in range(0,4):
            dx_total = dx_total + abs(post_coords[point, 0] - self.pre_coords[point, 0])
            dy_total = dy_total + abs(post_coords[point, 1] - self.pre_coords[point, 1])
            
        dx_avg = dx_total/4.0
        dy_avg = dy_total/4.0     
        dt_avg = math.sqrt(dx_avg*dx_avg + dy_avg*dy_avg)
        
        # Apply metric - tanh, any value larger than 15mm will be 100% 
        # and before 15mm is exponential to 100%
        scaled_dt = dt_avg * 0.0184
        uncertainty = np.tanh(scaled_dt)
        
        return uncertainty
    
        
    
    
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution', default=200, type=int,
                        help="resolution of the contours. Controls the refinement of the curve (default: 400)")
    parser.add_argument('--delta', type=int, default=10,
                        help="maximum distance allowed between two points in the trim (default: 10)")
    parser.add_argument('--eps', type=float, default=4.,
                        help="distance threshold when comparing points between the two contours (default: 5.0)")
    parser.add_argument('--height', type=float, default=0.17)
    parser.add_argument('--traj-time', type=float, default=20.)
    parser.add_argument('--traj-type', type=str, default='cut')
    parser.add_argument('--transparent', action='store_true')
    parser.add_argument('--feedback', action='store_true')
    parser.add_argument('--manual', action='store_true')
    parser.add_argument('--load-image', type=str, default=None)
    args = parser.parse_args()
    kwargs = vars(args)

    planner = PlanTrajectory(**kwargs)
    
    camera = cv2.VideoCapture(-1)
    for idx in range(300):
        ret, img = camera.read()

    img_dir = './test_images'
    img = cv2.imread(os.path.join(img_dir, '0.jpeg'))
    
    planned_traj = planner.plan(img)