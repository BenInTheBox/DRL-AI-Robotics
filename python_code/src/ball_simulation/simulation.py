from typing import Tuple

import json
import numpy as np
from numpy import ndarray

from ..constants import DT


class BallSimulation:

    def __init__(self):
        super(BallSimulation, self).__init__()
        with open('src/data/ball.json') as json_file:
            data = json.load(json_file)
            print(data)

        self.a = np.ndarray = np.array(data['phy_i_coef'])
        self.a[:, :2] = self.a[:, :2] * data['s_scaling']
        self.a[:, 2:] = self.a[:, 2:] * data['sin_scaling']
        self.b = np.ndarray = np.array(data['phy_i_bias'])

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
        self.d_x += np.dot(self.a, x) + self.b
        self.x += self.d_x * DT