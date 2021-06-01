from abc import ABC
from typing import Callable, Tuple

import gym
import numpy as np
import torch

from gym import spaces
from .simulation import BbSimulation, loss
from ..DDPG.controllers import BallController
from ..constants import MAX_X, DT, BALL_ERROR_SCALING, BALL_D_ERROR_SCALING, BALL_INTEGRAL_ERROR_SCALING, MAX_ANGLE


# Trajectory generation
def target_gen():
    target = np.zeros((2,))
    while True:
        target[0] += np.random.normal(-target[0] / 2000, MAX_X / 25., 1)
        target[1] += np.random.normal(-target[1] / 2000, MAX_X / 25., 1)
        yield target


def trajectory_gen(length: int) -> np.ndarray:
    gen = target_gen()
    trajectory = np.zeros((2, length))
    for i in range(length):
        trajectory[:, i] = next(gen)
    return trajectory


# Reward functions
def linear_reward(error: np.ndarray, d_error: np.ndarray, target: np.ndarray) -> float:
    # if (-d_error * DT < error).any():
    return - np.sum(np.sign(error) * d_error * BALL_D_ERROR_SCALING)


def linear_penality_reward(error: np.ndarray, d_error: np.ndarray, target: np.ndarray) -> float:
    # if (-d_error * DT < error).any():
    return - np.sum(np.sign(error) * d_error * BALL_D_ERROR_SCALING) - np.sum(np.abs(target / MAX_ANGLE))


# Environement "trait"
class BBEnvBasis(gym.Env, BbSimulation, ABC):
    """
    Abstract class for the environement, it ensure having the same environement for the commun methods and allows easy modifications
    """

    def __init__(self):
        super(BBEnvBasis, self).__init__()

        # Actions, State, Observation
        self.action_space = None
        self.state_space = spaces.Tuple(spaces=[
            spaces.Box(low=-0.8 * MAX_X, high=0.8 * MAX_X, shape=(2,), dtype=np.float32),  # Target position
            spaces.Box(low=-0.8 * MAX_X, high=0.8 * MAX_X, shape=(2,), dtype=np.float32),  # Ball position
            spaces.Box(low=-MAX_X / 5., high=MAX_X / 5., shape=(2,), dtype=np.float32),  # Ball speed
        ])
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,))

        self.state = None
        self.observation = None

        # Traing parameters
        self.max_iter: int = int(10. // DT)
        self.iter: int = 0
        self.reward: Callable = linear_penality_reward

        self.reset()

    def step(self, action):
        return NotImplemented

    def reset(self):
        self.reset_bb()
        self.state = self.state_space.sample()
        self.observation = np.array([0.] * 6, dtype=np.float32)

        self.ball.x = self.state[1]
        self.ball.d_x = self.state[2]

        self.observe()
        self.iter = 0

        return self.scaled_obs()

    def observe(self):
        obs = np.zeros_like(self.observation)
        obs[0:2] = self.state[1] - self.state[0]
        obs[2:4] = (obs[0:2] - self.observation[0:2]) / DT
        obs[4] = max(min(self.observation[4] + obs[0] * DT, MAX_X / 2), - MAX_X / 2)
        obs[5] = max(min(self.observation[5] + obs[1] * DT, MAX_X / 2), - MAX_X / 2)

        self.observation = obs

    def scaled_obs(self):
        observation = np.zeros_like(self.observation)
        observation[0:2] = self.observation[0:2] * BALL_ERROR_SCALING
        observation[2:4] = self.observation[2:4] * BALL_D_ERROR_SCALING
        observation[4:6] = self.observation[4:6] * BALL_INTEGRAL_ERROR_SCALING
        return observation

    def render(self, mode='human'):
        pass


class BBEnv(BBEnvBasis):
    """
    Ball balancer environement, obs -> motors angle
    """

    def __init__(self):
        super(BBEnv, self).__init__()

        # Actions, State, Observation
        self.action_space = spaces.Box(low=-1., high=1., shape=(2,), dtype=np.float32)

        # Traing parameters
        self.max_iter: int = int(10. // DT)
        self.iter: int = 0
        self.reward: Callable = linear_reward

        self.reset()

        print(self.state)

    def step(self, action):
        self.step_bb(action * MAX_ANGLE)
        self.state = (self.state[0], self.ball.x, self.ball.d_x)
        self.observe()
        reward: float = self.reward(self.observation[0:2], self.observation[2:4], action * MAX_ANGLE)

        self.iter += 1
        done: bool = (self.iter >= self.max_iter) or np.any(np.abs(self.ball.x) > 2 * MAX_X)
        info = {'state': self.state, 'observation': self.observation}

        return self.scaled_obs(), reward, done, info

    def simulate(self, model, target_trajectory: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        self.reset()
        trajectory = np.zeros_like(target_trajectory)
        error = np.zeros_like(target_trajectory)
        u = np.zeros_like(target_trajectory)
        angle = np.zeros_like(target_trajectory)
        error[:, 0] = target_trajectory[:, 0]
        integral: np.ndarray = error[:, 0] * DT

        for i in range(1, target_trajectory.shape[1]):
            trajectory[:, i] = self.state[1]
            error[:, i] = self.ball.x - target_trajectory[:, i]
            d_error: np.ndarray = (error[:, i] - error[:, i - 1]) / DT
            integral = integral + error[:, i] * DT

            integral[0] = max(min(integral[0], MAX_X / 2), - MAX_X / 2)
            integral[1] = max(min(integral[1], MAX_X / 2), - MAX_X / 2)

            angle[0, i] = self.motor_x.angle
            angle[1, i] = self.motor_y.angle

            obs = np.array(
                [error[0, i] * BALL_ERROR_SCALING, error[1, i] * BALL_ERROR_SCALING, d_error[0] * BALL_D_ERROR_SCALING,
                 d_error[1] * BALL_D_ERROR_SCALING, integral[0] * BALL_INTEGRAL_ERROR_SCALING,
                 integral[1] * BALL_INTEGRAL_ERROR_SCALING]
            )
            # print(obs)

            u[:, i] = model.act(torch.as_tensor(obs, dtype=torch.float32))

            self.step(u[:, i])

        u = MAX_ANGLE * u

        return trajectory, error, u, angle, loss(error)


class BBEnvPid(BBEnvBasis):
    """
    Ball balancer environement for dynamic PID, obs -> pid weights
    """

    def __init__(self):
        super(BBEnvPid, self).__init__()

        # Actions, State, Observation
        self.action_space = spaces.Box(low=0., high=0.33, shape=(6,), dtype=np.float32)

        self.state = None
        self.observation = None

        # Traing parameters
        self.max_iter: int = int(10. // DT)
        self.iter: int = 0
        self.reward: Callable = linear_penality_reward

        self.reset()

        print(self.state)

    def step(self, action):
        obs = self.scaled_obs()
        u_x = np.sum(-action[[0, 2, 4]] * obs[[0, 2, 4]])
        u_y = np.sum(-action[[1, 3, 5]] * obs[[1, 3, 5]])
        angles = np.tanh(np.array([u_x, u_y])) * MAX_ANGLE
        self.step_bb(angles)
        self.state = (self.state[0], self.ball.x, self.ball.d_x)
        self.observe()
        reward: float = self.reward(self.observation[0:2], self.observation[2:4], angles)

        self.iter += 1
        done: bool = (self.iter >= self.max_iter) or np.any(np.abs(self.ball.x) > 2 * MAX_X)
        info = {'state': self.state, 'observation': self.observation}

        return self.scaled_obs(), reward, done, info

    def simulate(self, model, target_trajectory: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
        self.reset()
        trajectory = np.zeros_like(target_trajectory)
        error = np.zeros_like(target_trajectory)
        u = np.zeros_like(target_trajectory)
        pid_kp = np.zeros_like(target_trajectory)
        pid_kd = np.zeros_like(target_trajectory)
        pid_ki = np.zeros_like(target_trajectory)
        angle = np.zeros_like(target_trajectory)
        error[:, 0] = target_trajectory[:, 0]
        integral: np.ndarray = error[:, 0] * DT

        for i in range(1, target_trajectory.shape[1]):
            trajectory[:, i] = self.state[1]
            error[:, i] = self.ball.x - target_trajectory[:, i]
            d_error: np.ndarray = (error[:, i] - error[:, i - 1]) / DT
            integral = integral + error[:, i] * DT

            integral[0] = max(min(integral[0], MAX_X / 2), - MAX_X / 2)
            integral[1] = max(min(integral[1], MAX_X / 2), - MAX_X / 2)

            angle[0, i] = self.motor_x.angle
            angle[1, i] = self.motor_y.angle

            obs = np.array(
                [error[0, i], error[1, i], d_error[0], d_error[1], integral[0], integral[1]]
            )
            self.observation = obs
            # print(obs)

            pid_weights = model.act(torch.as_tensor(self.scaled_obs(), dtype=torch.float32))
            pid_kp[0, i] = -pid_weights[0]
            pid_kp[1, i] = -pid_weights[1]
            pid_kd[0, i] = -pid_weights[2]
            pid_kd[1, i] = -pid_weights[3]
            pid_ki[0, i] = -pid_weights[4]
            pid_ki[1, i] = -pid_weights[5]

            self.step(pid_weights)
            u_x = np.sum(- pid_weights[[0, 2, 4]] * obs[[0, 2, 4]])
            u_y = np.sum(- pid_weights[[1, 3, 5]] * obs[[1, 3, 5]])
            u[:, i] = np.tanh(np.array([u_x, u_y])) * MAX_ANGLE

        return trajectory, error, u, angle, loss(error), pid_kp, pid_kd, pid_ki


# Benchmark Environement
class BenchmarkEvaluator(BBEnv):

    def __init__(self, target_trajectory: np.ndarray):
        super(BenchmarkEvaluator, self).__init__()

        self.target_trajectory: np.ndarray = target_trajectory

    def simulate(self, model: BallController) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        self.reset()
        trajectory = np.zeros_like(self.target_trajectory)
        error = np.zeros_like(self.target_trajectory)
        u = np.zeros_like(self.target_trajectory)
        angle = np.zeros_like(self.target_trajectory)
        error[:, 0] = self.target_trajectory[:, 0]
        integral: np.ndarray = error[:, 0] * DT

        for i in range(1, self.target_trajectory.shape[1]):
            trajectory[:, i] = self.state[1]
            error[:, i] = self.ball.x - self.target_trajectory[:, i]
            d_error: np.ndarray = (error[:, i] - error[:, i - 1]) / DT
            integral += error[:, i] * DT

            integral[0] = max(min(integral[0], MAX_X / 2), - MAX_X / 2)
            integral[1] = max(min(integral[1], MAX_X / 2), - MAX_X / 2)

            angle[0, i] = self.motor_x.angle
            angle[1, i] = self.motor_y.angle

            u[0, i] = model.step(error[0, i], d_error[0], integral[0])
            u[1, i] = model.step(error[1, i], d_error[1], integral[1])

            self.step(u[:, i] / MAX_ANGLE)

        return trajectory, error, u, angle, loss(error)

    def evaluate(self, model: BallController) -> float:
        _, _, _, _, loss = self.simulate(model)
        # print("Loss", loss)
        return loss
