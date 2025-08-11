import pygame
import numpy as np
    

class Joystick(object):

    def __init__(self):
        pygame.init()
        self.gamepad = pygame.joystick.Joystick(0)
        self.gamepad.init()
        self.toggle = False
        self.action = None
        self.A_pressed = False
        self.B_pressed = False

        # some constants
        self.step_size_l = 0.01
        self.step_size_a = 0.2 * np.pi / 4
        self.step_time = 0.01
        self.deadband = 0.1

    def getInput(self):
        pygame.event.get()
        toggle_angular = self.gamepad.get_button(4)
        toggle_linear = self.gamepad.get_button(5)
        self.A_pressed = self.gamepad.get_button(0)
        self.B_pressed = self.gamepad.get_button(1)
        if not self.toggle and toggle_angular:
            self.toggle = True
        elif self.toggle and toggle_linear:
            self.toggle = False
        return self.getEvent()

    def getEvent(self):
        z1 = self.gamepad.get_axis(0)
        z2 = self.gamepad.get_axis(1)
        z3 = self.gamepad.get_axis(4)
        z = [-z1, z2, -z3]
        for idx in range(len(z)):
            if abs(z[idx]) < self.deadband:
                z[idx] = 0.0
        stop = self.gamepad.get_button(7)
        X_pressed = self.gamepad.get_button(2)
        B_pressed = self.gamepad.get_button(1)
        A_pressed = self.gamepad.get_button(0)
        return tuple(z), A_pressed, B_pressed, X_pressed, stop

    def getAction(self, z):
        if self.toggle:
            self.action = (0, 0, 0, np.round(self.step_size_a * -z[1], 2), np.round(self.step_size_a * -z[0], 2), np.round(self.step_size_a * -z[2], 2))
        else:
            self.action = (np.round(self.step_size_l * z[1], 2), np.round(self.step_size_l * -z[0], 2), np.round(self.step_size_l * z[2], 2), 0, 0, 0)

