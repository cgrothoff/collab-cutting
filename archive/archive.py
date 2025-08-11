def slice(self, **kwargs):
        ''' slice into given number of slices '''
        meat_thresholds = np.array(json.load(open("./config/meat.json", "r")))
        meat_img = cv2.inRange(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB), meat_thresholds[0], meat_thresholds[1])
        meat_contour = get_edges(meat_img)
        meat_contour = np.concatenate((meat_contour, [meat_contour[0]]), axis=0)
        
        # get rectangle around meat
        x,y,w,h = cv2.boundingRect(meat_contour)
        
        # show meat box for debugging
        cv2.rectangle(self.image, (x, y), (x + w, y + h), (255,0,0), 4)
        cv2.imshow('test', self.image)
        cv2.waitKey(1000)
        
        # make the slicing path
        n_cuts = 2
        n_points = 10
        square = np.zeros(((n_cuts-1)*n_points, 2))
        x_coords = np.linspace(0, 1, n_cuts + 1)[1:-1]
        for idx, xc in enumerate(x_coords):
            # if idx % 2 == 0:
            #     square[idx*n_points:(idx+1)*n_points, :] = np.column_stack((np.ones(n_points)*xc, np.linspace(0, 1, n_points)))
            # else:
            #     square[idx*n_points:(idx+1)*n_points, :] = np.column_stack((np.ones(n_points)*xc, np.linspace(1, 0, n_points)))
            square[idx*n_points:(idx+1)*n_points, :] = np.column_stack((np.ones(n_points)*xc, np.linspace(0, 1, n_points)))

        # scale to the bounding box
        square[:, 0] = w * square[:,0] + x
        square[:, 1] = 1.4 * h * square[:,1] + y - .25*h
        slice = square

        # print slice for debugging
        slice_int = slice.astype(int)
        debug_img = cv2.drawContours(self.image, [slice_int.reshape((-1, 1, 2))], -1, [0, 0, 255], 2)
        cv2.imshow("test", debug_img)
        cv2.waitKey(1000)
        
        # ask for human feedback
        if self.gui:
            updated_slice = self.show_interface(slice_int)
            
        # get the final trajectory
        waypoints = transform(updated_slice, self.height)
        
        traj_times = np.linspace(0., self.traj_time, waypoints.shape[0])
        trajectory = np.column_stack((traj_times, waypoints))
        
        return trajectory
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    