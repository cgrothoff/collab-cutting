import pygame
import numpy as np
import matplotlib.pyplot as plt
import time
import os


# colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)


class GUI:
    def __init__(self, image, trajectory, type, point_radius=6, line_width=2):
        pygame.init()
        
        self.image = image
        self.type = type
        self.trajectory = trajectory
        self.line_width = line_width
        self.point_radius = point_radius
        self.num_points = 0
        if self.trajectory is not None:
            self.num_points = len(trajectory)
        
        height, width, _ = self.image.shape
        self.screen_size = (height, width)
                
        self.screen = pygame.display.set_mode(self.screen_size)
        pygame.display.set_caption("Planned Trajectory")
        
        self.surface = pygame.surfarray.make_surface(self.image)

        self.button_surface = pygame.Surface((150, 50), )
        font = pygame.font.Font(None, 24)
        self.text = font.render("Approve", True, (0, 0, 0))

        self.button_rect = pygame.Rect(245, 420, 150, 50)
        self.text_rect = self.text.get_rect(center=(320, 445))
        
        pygame.draw.rect(self.button_surface, (0, 0, 0), (0, 0, 150, 50))
        pygame.draw.rect(self.button_surface, (0, 200, 0), (1, 1, 148, 48))
            
        self.update_screen()
        
    def draw_line(self, start_pos, end_pos, c=BLACK):
        pygame.draw.line(self.screen, c, start_pos, end_pos, self.line_width)
        
    def draw_point(self, pos, c=BLUE):
        pygame.draw.circle(self.screen, c, pos, self.point_radius)

    def get_closer_point(self, mouse_pos):
        for idx in range(self.num_points):
            if pygame.math.Vector2(tuple(self.trajectory[idx, :])).distance_to(mouse_pos) < self.point_radius:
                return idx
        return None
    
    def recalculate_curve(self, mouse_pos):
        # fit the new point along the curve
        if self.trajectory is not None:
            distances = np.linalg.norm(np.array(mouse_pos) - self.trajectory, axis=1)
            nearest_point_idx = np.argsort(distances)[0]
            if nearest_point_idx != 0:
                nearest_point_idx += 1
        
            self.trajectory = np.insert(self.trajectory, [nearest_point_idx], mouse_pos, axis=0)
        elif self.trajectory is None:
            self.trajectory = np.array([mouse_pos])
        else:
            self.trajectory = np.row_stack((self.trajectory, np.array(mouse_pos)), axis=0)
             
    def update_trajectory(self, mouse_pos, manipulated_point_idx, add=True):
        if add:
            if manipulated_point_idx is not None:
                self.trajectory[manipulated_point_idx, :] = pygame.mouse.get_pos()
            else:
                self.recalculate_curve(mouse_pos)
        else:
            self.trajectory = np.delete(self.trajectory, manipulated_point_idx, axis=0)
        self.num_points = len(self.trajectory)
        
    def render(self):
        manipulated_point_idx = None
        running = True
            
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1: 
                        # check if mouse click is near any of the exisiting points
                        mouse_pos = pygame.mouse.get_pos()
                        manipulated_point_idx = self.get_closer_point(mouse_pos)
                        
                        if manipulated_point_idx is None and self.type != 'transparent':
                            self.update_trajectory(mouse_pos, manipulated_point_idx)
                    
                    elif event.button == 3 and self.type != 'transparent':
                        mouse_pos = pygame.mouse.get_pos()
                        manipulated_point_idx = self.get_closer_point(mouse_pos)
                        
                        if manipulated_point_idx is not None:
                            self.update_trajectory(None, manipulated_point_idx, add=False)
                        
                elif event.type == pygame.MOUSEBUTTONUP:
                    manipulated_point_idx = None
                    
                else:
                    mouse_pos = pygame.mouse.get_pos()
                    # check if over the button
                    if self.button_rect.collidepoint(mouse_pos):
                        pygame.draw.rect(self.button_surface, (0, 0, 0), (0, 0, 150, 50))
                        pygame.draw.rect(self.button_surface, (50, 200, 50), (1, 1, 148, 48))
                        
            if self.type == 'feedback' or self.type == 'manual':
                # move the point
                if pygame.mouse.get_pressed()[0] and manipulated_point_idx is not None:
                    self.update_trajectory(mouse_pos, manipulated_point_idx)
                    
            # clear the screen
            self.screen.fill(WHITE)
            self.update_screen()
        
        savedir = './debug'
        if not os.path.exists(savedir):
            os.makedirs(savedir)
        pygame.image.save(self.screen, '{}/{}.jpg'.format(savedir, time.strftime('%m_%d_%y_%H_%M_%S')))
        pygame.display.quit()
            
    def update_screen(self):
        self.screen.blit(self.surface, (0, 0))
        if self.trajectory is not None:
            # Draw the lines and points
            for idx2 in range(self.num_points - 1):
                self.draw_line(self.trajectory[idx2, :], self.trajectory[idx2 + 1, :])
                self.draw_point(self.trajectory[idx2, :])
            self.draw_point(self.trajectory[-1, :], WHITE)
        
        if self.type == 'transparent':
            
            self.screen.blit(self.button_surface, (self.button_rect.x, self.button_rect.y))
            self.screen.blit(self.text, self.text_rect)
                
        pygame.display.flip()
        
    def get_trajectory(self):
        return self.trajectory
    
    def quit(self):
        pygame.quit()
        
        
if __name__ == '__main__':
    image = np.ones((640, 480, 3)) * 255
    image = image.astype(np.uint8)
    
    num_points = 4
    random_waypints_x = np.random.uniform(0, image.shape[0], size=num_points).astype(dtype=np.int)
    random_waypints_y = np.random.uniform(0, image.shape[1], size=num_points).astype(dtype=np.int)
    random_waypints = np.column_stack((random_waypints_x, random_waypints_y))

    gui = GUI(image, random_waypints)
    
    gui.render()

