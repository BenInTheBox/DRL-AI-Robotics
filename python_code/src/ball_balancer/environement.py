from abc import ABC
from typing import Callable, Tuple

import gym
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt

from gym import spaces
from .simulation import BbSimulation, loss
from ..DDPG.controllers import BallController
from ..constants import MAX_X, DT, BALL_ERROR_SCALING, BALL_D_ERROR_SCALING, BALL_INTEGRAL_ERROR_SCALING, MAX_ANGLE, \
    FILTERING_PERIOD, BALL_MAX_INTEGRAL
from mpl_toolkits import mplot3d
from matplotlib import cm
from matplotlib.ticker import LinearLocator


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
def linear_de_reward(error: np.ndarray, d_error: np.ndarray, target: np.ndarray, w) -> float:
    if (d_error[0] * DT + error[0]) * np.sign(error[0]) < 0.:
        d_error[0] = - error[0] - d_error[0] * DT
    if (d_error[1] * DT + error[1]) * np.sign(error[1]) < 0.:
        d_error[1] = - error[1] - d_error[1] * DT
    return np.tanh(- np.sum(np.sign(error) * d_error * DT * BALL_D_ERROR_SCALING * w))


def linear_de_penality_reward(error: np.ndarray, d_error: np.ndarray, target: np.ndarray, w) -> float:
    # if (-d_error * DT < error).any():
    return np.tanh(
        0.5 * w * - np.sum(np.sign(error) * d_error * BALL_D_ERROR_SCALING) - np.sum(np.abs(target / MAX_ANGLE)))


def linear_e_reward(error: np.ndarray, d_error: np.ndarray, target: np.ndarray, w) -> float:
    return np.tanh(1. - (np.sum(np.abs(error)) * 30.) ** w)


def quadratic_e_reward(error: np.ndarray, d_error: np.ndarray, target: np.ndarray, w) -> float:
    return np.tanh(1. - (np.sum(np.float_power(error * 10., 2) ** w)))


def quadratic_e_reward_penality(error: np.ndarray, d_error: np.ndarray, target: np.ndarray, w) -> float:
    return (2. * w + np.tanh(1. - (np.sum(np.float_power(error * 10., 2)))) - w * np.sum(np.abs(target) / MAX_ANGLE)) / (
                1. + 2. * w)


def test_reward():
    error = np.arange(-MAX_X, 0., 0.001)
    w = np.arange(0.3, 0.9, 0.1)
    target = np.arange(0., MAX_ANGLE, 1)
    d_error = np.arange(-MAX_X, MAX_X, 0.002)
    x_error, y_error = np.meshgrid(error, error)

    z = [np.array([linear_e_reward(np.array([x, x]), 0., 0., weight) for x in error]) for weight in w]
    Z = np.array([[linear_e_reward(np.array([x, y]), 0., 0., 0.6) for x, y in zip(x_row, y_row)] for x_row, y_row in
                  zip(x_error, y_error)])

    plt.figure()
    for weight, ts in zip(w, z):
        plt.plot(error * 2, ts, label='w = ' + str(weight))
    plt.title('Linear error reward 1D')
    plt.xlabel('abs(error x) + abs(error y)')
    plt.ylabel('reward')
    plt.legend()
    plt.show()
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.view_init(30, 120)
    # Plot the surface.
    surf = ax.plot_surface(x_error, y_error, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    ax.set_xlabel('$$', fontsize=20)
    ax.set_xlabel('$error_{x,t}$', fontsize=20)
    ax.set_ylabel('$error_{y,t}$', fontsize=20)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

    w = np.arange(0.8, 1.5, 0.2)
    z = [np.array([quadratic_e_reward(np.array([x, x]), 0., 0., weight) for x in error]) for weight in w]
    Z = np.array([[quadratic_e_reward(np.array([x, y]), 0., 0., 1.) for x, y in zip(x_row, y_row)] for x_row, y_row in
                  zip(x_error, y_error)])

    plt.figure()
    for weight, ts in zip(w, z):
        plt.plot(error * 2, ts, label='w = ' + str(weight))
    plt.title('Power error reward 1D')
    plt.xlabel('abs(error x) + abs(error y)')
    plt.ylabel('reward')
    plt.legend()
    plt.show()
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.view_init(30, 120)
    # Plot the surface.
    surf = ax.plot_surface(x_error, y_error, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    ax.set_xlabel('$error_{x,t}$', fontsize=20)
    ax.set_ylabel('$error_{y,t}$', fontsize=20)
    ax.set_zlabel('$r_{t}$', fontsize=20)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

    x_error_sum, angle_sum = np.meshgrid(error, target)
    Z = np.array(
        [[quadratic_e_reward_penality(np.array([x, x]), 0., np.array([y, y]), 1) for x, y in zip(x_row, y_row)] for
         x_row, y_row in
         zip(x_error_sum, angle_sum)])
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.view_init(30, 120)
    # Plot the surface.
    surf = ax.plot_surface(x_error_sum, angle_sum, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    ax.set_xlabel(r'$\sum_{}^{}errors$', fontsize=20)
    ax.set_ylabel(r'$\sum_{}^{}angles$', fontsize=20)
    ax.set_zlabel('$r_{t}$', fontsize=20)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

    e = np.array([0.5, 0.5])
    w = np.arange(5, 35, 10)
    z = [np.array([linear_de_reward(e, np.array([x, x]), 0., weight) for x in d_error]) for weight in w]

    plt.figure()
    for weight, ts in zip(w, z):
        plt.plot(d_error * 2., ts, label='w = ' + str(weight))
    plt.xlabel('d error x + d error y')
    plt.ylabel('reward')
    plt.title('Linear d error reward 1D')
    plt.legend()
    plt.show()


# Environement creation
def env_fn_gen(env, reward_fn, reward_weight):
    def create_env():
        return env(reward_func=reward_fn, reward_w=reward_weight)

    return create_env


# Environement "trait"
class BBEnvBasis(gym.Env, BbSimulation, ABC):
    """
    Abstract class for the environement, it ensure having the same environement for the commun methods and allows easy modifications
    """

    def __init__(self, reward_func=linear_e_reward, reward_w=0.5):
        super(BBEnvBasis, self).__init__()

        # Actions, State, Observation
        self.action_space = None
        self.state_space = spaces.Tuple(spaces=[
            spaces.Box(low=-0.8 * MAX_X, high=0.8 * MAX_X, shape=(2,), dtype=np.float32),  # Target position
            spaces.Box(low=-0.8 * MAX_X, high=0.8 * MAX_X, shape=(2,), dtype=np.float32),  # Ball position
            spaces.Box(low=-MAX_X / 30., high=MAX_X / 30., shape=(2,), dtype=np.float32),  # Ball speed
        ])
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,))

        self.state = None
        self.observation = None
        self.ema: np.ndarray = np.array([0., 0.])
        self.dema: np.ndarray = np.array([0., 0.])
        self.real_error: np.ndarray = np.array([0., 0.])
        self.real_d_error: np.ndarray = np.array([0., 0.])
        self.alpha: float = 2. / (1. + FILTERING_PERIOD)

        # Traing parameters
        self.max_iter: int = int(10. // DT)
        self.iter: int = 0
        self.reward: Callable = reward_func
        self.w = reward_w

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
        self.ema = self.state[1]
        self.dema = self.state[1]
        self.real_error = np.array([0., 0.])
        self.real_d_error = np.array([0., 0.])

        return self.scaled_obs()

    def test_reset(self):
        self.reset_bb()
        zero_arr = np.zeros_like(self.state_space.sample()[0])
        self.state = (zero_arr, zero_arr, zero_arr)
        self.observation = np.array([0.] * 6, dtype=np.float32)

        self.ball.x = self.state[1]
        self.ball.d_x = self.state[2]

        self.observe()
        self.iter = 0
        self.ema = self.state[1]
        self.dema = self.state[1]
        self.real_error = np.array([0., 0.])
        self.real_d_error = np.array([0., 0.])

        return self.scaled_obs()

    def observe(self):
        obs = np.zeros_like(self.observation)
        ema = self.alpha * self.state[1] + (1 - self.alpha) * self.ema
        dema = self.alpha * ema + (1 - self.alpha) * self.dema
        obs[0:2] = dema - self.state[0]
        real_error = self.state[1] - self.state[0]
        self.real_d_error = (real_error - self.real_error) / DT
        self.real_error = real_error
        obs[2:4] = (obs[0:2] - self.observation[0:2]) / DT
        obs[4] = max(min(self.observation[4] + obs[0] * DT, BALL_MAX_INTEGRAL), - BALL_MAX_INTEGRAL)
        obs[5] = max(min(self.observation[5] + obs[1] * DT, BALL_MAX_INTEGRAL), - BALL_MAX_INTEGRAL)

        self.observation = obs
        self.ema = ema
        self.dema = dema

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

    def __init__(self, reward_func=linear_e_reward, reward_w=0.5):
        super(BBEnv, self).__init__()

        # Actions, State, Observation
        self.action_space = spaces.Box(low=-1., high=1., shape=(2,), dtype=np.float32)

        # Traing parameters
        self.max_iter: int = int(10. // DT)
        self.iter: int = 0
        self.reward: Callable = reward_func
        self.w: float = reward_w

        self.reset()

        print(self.state)

    def step(self, action):
        self.step_bb(action * MAX_ANGLE)
        self.state = (self.state[0], self.ball.x, self.ball.d_x)
        self.observe()
        reward: float = self.reward(self.real_error, self.real_d_error, action * MAX_ANGLE, self.w)

        self.iter += 1
        done: bool = (self.iter >= self.max_iter) or np.any(np.abs(self.ball.x) > 2 * MAX_X)
        info = {'state': self.state, 'observation': self.observation}

        return self.scaled_obs(), reward, done, info

    def simulate(self, model, target_trajectory: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        self.test_reset()
        trajectory = np.zeros_like(target_trajectory)
        error = np.zeros_like(target_trajectory)
        u = np.zeros_like(target_trajectory)
        angle = np.zeros_like(target_trajectory)
        error[:, 0] = target_trajectory[:, 0]
        integral: np.ndarray = error[:, 0] * DT

        for i in range(1, target_trajectory.shape[1]):
            trajectory[:, i] = self.state[1]
            ema = self.alpha * self.state[1] + (1 - self.alpha) * self.ema
            dema = self.alpha * ema + (1 - self.alpha) * self.dema
            error[:, i] = dema - target_trajectory[:, i]
            d_error: np.ndarray = (error[:, i] - error[:, i - 1]) / DT
            integral = integral + error[:, i] * DT

            integral[0] = max(min(integral[0], BALL_MAX_INTEGRAL), - BALL_MAX_INTEGRAL)
            integral[1] = max(min(integral[1], BALL_MAX_INTEGRAL), - BALL_MAX_INTEGRAL)

            angle[0, i] = self.motor_x.angle
            angle[1, i] = self.motor_y.angle

            obs = np.array(
                [error[0, i] * BALL_ERROR_SCALING, error[1, i] * BALL_ERROR_SCALING, d_error[0] * BALL_D_ERROR_SCALING,
                 d_error[1] * BALL_D_ERROR_SCALING, integral[0] * BALL_INTEGRAL_ERROR_SCALING,
                 integral[1] * BALL_INTEGRAL_ERROR_SCALING]
            )
            # print(obs)

            u[:, i] = model.act(torch.as_tensor(obs, dtype=torch.float32))
            self.ema = ema
            self.dema = dema

            self.step(u[:, i])

        u = MAX_ANGLE * u

        return trajectory, error, u, angle, loss(error)


class BBEnvPid(BBEnvBasis):
    """
    Ball balancer environement for dynamic PID, obs -> pid weights
    """

    def __init__(self, reward_func=linear_e_reward, reward_w=0.5):
        super(BBEnvPid, self).__init__()

        # Actions, State, Observation
        self.action_space = spaces.Box(low=0., high=2.5, shape=(6,), dtype=np.float32)

        self.state = None
        self.observation = None

        # Traing parameters
        self.max_iter: int = int(10. // DT)
        self.iter: int = 0
        self.reward: Callable = reward_func
        self.w: float = reward_w

        self.reset()

        self.last_angles = np.array([0., 0.])

        print(self.state)

    def step(self, action):
        obs = self.scaled_obs()
        u_x = np.sum(action[[0, 2, 4]] * obs[[0, 2, 4]])
        u_y = np.sum(action[[1, 3, 5]] * obs[[1, 3, 5]])
        angles = np.tanh(np.array([u_x, u_y])) * MAX_ANGLE
        self.last_angles = angles
        self.step_bb(angles)
        self.state = (self.state[0], self.ball.x, self.ball.d_x)
        self.observe()
        reward: float = self.reward(self.real_error, self.real_d_error, angles, self.w)

        self.iter += 1
        done: bool = (self.iter >= self.max_iter) or np.any(np.abs(self.ball.x) > 2 * MAX_X)
        info = {'state': self.state, 'observation': self.observation}

        return self.scaled_obs(), reward, done, info

    def simulate(self, model, target_trajectory: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
        self.test_reset()
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
            ema = self.alpha * self.state[1] + (1 - self.alpha) * self.ema
            dema = self.alpha * ema + (1 - self.alpha) * self.dema
            error[:, i] = dema - target_trajectory[:, i]
            d_error: np.ndarray = (error[:, i] - error[:, i - 1]) / DT
            integral = integral + error[:, i] * DT

            integral[0] = max(min(integral[0], BALL_MAX_INTEGRAL), - BALL_MAX_INTEGRAL)
            integral[1] = max(min(integral[1], BALL_MAX_INTEGRAL), - BALL_MAX_INTEGRAL)

            angle[0, i] = self.motor_x.angle
            angle[1, i] = self.motor_y.angle

            obs = np.array(
                [error[0, i], error[1, i], d_error[0], d_error[1], integral[0], integral[1]]
            )
            self.observation = obs
            scaled_obs = self.scaled_obs()
            # print(obs)

            pid_weights = model.act(torch.as_tensor(scaled_obs, dtype=torch.float32))
            pid_kp[0, i] = pid_weights[0]
            pid_kp[1, i] = pid_weights[1]
            pid_kd[0, i] = pid_weights[2]
            pid_kd[1, i] = pid_weights[3]
            pid_ki[0, i] = pid_weights[4]
            pid_ki[1, i] = pid_weights[5]

            self.step(pid_weights)
            u_x = np.sum(pid_weights[[0, 2, 4]] * scaled_obs[[0, 2, 4]])
            u_y = np.sum(pid_weights[[1, 3, 5]] * scaled_obs[[1, 3, 5]])
            u[:, i] = np.tanh(np.array([u_x, u_y])) * MAX_ANGLE
            self.ema = ema
            self.dema = dema

        return trajectory, error, u, angle, loss(error), pid_kp, pid_kd, pid_ki


# Benchmark Environement
class BenchmarkEvaluator(BBEnv):

    def __init__(self, target_trajectory: np.ndarray):
        super(BenchmarkEvaluator, self).__init__()

        self.target_trajectory: np.ndarray = target_trajectory

    def simulate(self, model: BallController, test=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        if test:
            self.test_reset()
        else:
            self.reset()
        trajectory = np.zeros_like(self.target_trajectory)
        error = np.zeros_like(self.target_trajectory)
        u = np.zeros_like(self.target_trajectory)
        angle = np.zeros_like(self.target_trajectory)
        error[:, 0] = self.target_trajectory[:, 0]
        integral: np.ndarray = error[:, 0] * DT

        for i in range(1, self.target_trajectory.shape[1]):
            trajectory[:, i] = self.state[1]
            ema = self.alpha * self.state[1] + (1 - self.alpha) * self.ema
            dema = self.alpha * ema + (1 - self.alpha) * self.dema
            error[:, i] = dema - self.target_trajectory[:, i]
            error[:, i] = self.ball.x - self.target_trajectory[:, i]
            d_error: np.ndarray = (error[:, i] - error[:, i - 1]) / DT
            integral += error[:, i] * DT

            integral[0] = max(min(integral[0], BALL_MAX_INTEGRAL), - BALL_MAX_INTEGRAL)
            integral[1] = max(min(integral[1], BALL_MAX_INTEGRAL), - BALL_MAX_INTEGRAL)

            angle[0, i] = self.motor_x.angle
            angle[1, i] = self.motor_y.angle
            # print(error[0, i], d_error[0], integral[0])
            u[0, i] = model.step(error[0, i], d_error[0], integral[0])
            u[1, i] = model.step(error[1, i], d_error[1], integral[1])

            self.step(u[:, i] / MAX_ANGLE)

        return trajectory, error, u, angle, loss(error)

    def evaluate(self, model: BallController) -> float:
        _, _, _, _, loss = self.simulate(model)
        # print("Loss", loss)
        return loss
