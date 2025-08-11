from scipy.interpolate import interp1d
import numpy as np

class Trajectory():

    def __init__(self, traj):
        self.traj = np.copy(traj)
        self.n, self.m = self.traj.shape
        self.t_start = self.traj[0, 0]
        self.t_end = self.traj[-1, 0]
        self.total_time = self.t_end - self.t_start
        self.times = self.traj[:, 0]
        
        self.interpolators = []
        for idx in range(self.m):
            self.interpolators.append(interp1d(self.times, self.traj[:, idx], kind='linear'))
        
    def get_gripper_action(self, sample_time):
        if sample_time in self.times:
            closest_gripper_idx = np.where(self.times==sample_time)
        else:
            closest_gripper_idx = np.searchsorted(self.times, sample_time) - 1
            closest_gripper_idx = np.clip(closest_gripper_idx, 0, len(self.times))

        return self.gripper_state[closest_gripper_idx]    
    
    def get_waypoint(self, t):
        if t < 0.0:
            t = 0.0
        if t > self.t_end:
            t = self.t_end
        waypoint = np.array([0.] * self.m)
        for idx in range(self.m):
            waypoint[idx] = self.interpolators[idx](t)
        return waypoint[1:]
    
    def get_segment_orientation(self):
        curr = self.get_waypoint(self.t_start)
        next = self.get_waypoint(self.t_end)
        v = next - curr
        return np.arctan2(v[1], v[0])