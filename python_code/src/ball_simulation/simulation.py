import json
import numpy as np

from ..constants import DT, MAX_X


class BallSimulation:

    def __init__(self):
        super(BallSimulation, self).__init__()
        with open('src/data/ball.json') as json_file:
            data = json.load(json_file)
            print(data)

        self.a: np.ndarray = np.array(data['phy_i_coef'])
        self.a[:, :2] = self.a[:, :2] * data['s_scaling']
        self.a[:, 2:] = self.a[:, 2:] * data['sin_scaling']

        self.x: np.ndarray = np.array([0., 0.])
        self.d_x: np.ndarray = np.array([0., 0.])

    def reset_ball(self):
        """
        Reset method. Put the system in zero state
        """
        self.x: np.ndarray = np.array([0., 0.])
        self.d_x: np.ndarray = np.array([0., 0.])

    def step(self, angle_x: float, angle_y: float):
        """
        Compute the next environement state from the input
        :param angle_x: Current input
        :param angle_y: Current input
        """
        x = np.array([self.d_x[0], self.d_x[1], np.sin(np.pi / 180. * angle_x), np.sin(np.pi / 180. * angle_y)])
        self.d_x = self.d_x + np.dot(self.a, x)
        # self.d_x[self.x > MAX_X] = 0.  # MAX_X
        # self.d_x[self.x < -MAX_X] = 0.  # -MAX_X
        self.x = self.x + self.d_x * DT
        # self.x[self.x > MAX_X] = MAX_X
        # self.x[self.x < -MAX_X] = -MAX_X


class BallSimulation1D:

    def __init__(self):
        super(BallSimulation1D, self).__init__()
        with open('src/data/ball_1d.json') as json_file:
            data = json.load(json_file)
            print(data)

        self.a: np.ndarray = np.array(data['phy_i_coef'])
        self.a[0] = self.a[0] * data['s_scaling']
        self.a[1] = self.a[1] * data['sin_scaling']

        self.x: float = 0.
        self.d_x: float = 0.

    def reset_ball(self):
        """
        Reset method. Put the system in zero state
        """
        self.x: float = 0.
        self.d_x: float = 0.

    def step(self, angle: float):
        """
        Compute the next environement state from the input
        :param angle: Current input
        """
        x = np.array([self.d_x, np.sin(np.pi / 180. * angle)])
        self.d_x = self.d_x + np.dot(self.a, x)
        self.x = self.x + self.d_x * DT
