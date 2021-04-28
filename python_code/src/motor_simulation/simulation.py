from typing import Tuple

import torch
import numpy as np
from numpy import ndarray

from .neural_net_controller import MotorController
from .neural_net_motor import Plateau, PlateauPhy
from ..constants import DT


class MotorSimulation:

    def __init__(self, motor_type: str = 'phy'):
        super(MotorSimulation, self).__init__()
        self.motor: Plateau
        if motor_type == 'phy':
            self.motor = PlateauPhy()
        elif motor_type == 'nn':
            self.motor = torch.load('src/data/motor_nn.pt')

        self.dt: float = DT

        self.angle: float = 0.
        self.speed: float = 0.

    def reset_motor(self):
        """
        Reset method. Put the system in zero state
        """
        self.angle = 0.
        self.speed = 0.

    def step(self, u: float):
        """
        Compute the next environement state from the input
        :param u: Current input
        """
        self.speed = self.motor.step(u, self.speed)
        self.angle = max(-180., min(180., self.angle + self.speed * self.dt))


def loss(error: ndarray) -> float:
    return - np.sqrt(np.mean(np.power(error, 2)))


class ModelEvaluator:

    def __init__(self, target_trajectory: ndarray):
        super(ModelEvaluator, self).__init__()
        self.target_trajectory = target_trajectory
        self.dt = DT

        self.phy_model: MotorSimulation = MotorSimulation()
        self.nn_model: MotorSimulation = MotorSimulation(motor_type='nn')

    def phy_simulate(self, model: MotorController) -> Tuple[ndarray, ndarray, ndarray, float]:
        self.phy_model.reset_motor()
        trajectory = np.zeros_like(self.target_trajectory)
        error = np.zeros_like(self.target_trajectory)
        u = np.zeros_like(self.target_trajectory)
        error[0] = self.target_trajectory[0]
        integral: float = error[0] * self.dt

        for i in range(1, len(self.target_trajectory)):
            trajectory[i] = self.phy_model.angle
            error[i] = self.target_trajectory[i] - self.phy_model.angle
            d_error: float = (error[i] - error[i - 1]) / self.dt
            integral += error[i] * self.dt

            u[i] = model.step(error[i], d_error, integral)
            self.phy_model.step(u[i])

        return trajectory, error, u, loss(error)

    def nn_simulate(self, model: MotorController) -> Tuple[ndarray, ndarray, ndarray, float]:
        self.nn_model.reset_motor()
        trajectory = np.zeros_like(self.target_trajectory)
        error = np.zeros_like(self.target_trajectory)
        u = np.zeros_like(self.target_trajectory)
        error[0] = self.target_trajectory[0]
        integral: float = error[0] * self.dt

        for i in range(1, len(self.target_trajectory)):
            trajectory[i] = self.nn_model.angle
            error[i] = self.target_trajectory[i] - self.nn_model.angle
            d_error: float = (error[i] - error[i - 1]) / self.dt
            integral += error[i] * self.dt

            u[i] = model.step(error[i], d_error, integral)
            self.nn_model.step(u[i])

        return trajectory, error, u, loss(error)

    def evaluate(self, model: MotorController) -> float:
        _, _, _, loss = self.phy_simulate(model)
        return loss
