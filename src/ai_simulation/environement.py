import gym
import numpy as np

from gym import spaces
from .environement_constants import BBProps


class BBEnvRaw(gym.Env, BBProps):
    """
    Ball balancer environement
    """

    def __init__(self):
        super(BBEnvRaw, self).__init__()
        super(BBProps, self).__init__()
        self.compute_k()

        # Action
        self.action_space: spaces = spaces.Box(low=0, high=10, shape=(1, 2, 1), dtype=np.float64)  # [V1, V2]

    def step(self, action):
        # Call compute_dynamics
        pass

    def reset(self):
        pass

    def render(self, mode='human'):
        pass


class BBEnvPid(gym.Env, BBProps):
    """
    Ball balancer environement
    """

    def __init__(self):
        super(BBEnvPid, self).__init__()
        super(BBProps, self).__init__()
        self.compute_k()

        # Action
        self.action_space = spaces.Box(low=0, high=10, shape=(1, 2, 1), dtype=np.float64)  # [Kp, Kd, Ki]

    def step(self, action):
        # Calculer V1, V2
        # Call compute_dynamics
        pass

    def reset(self):
        pass

    def render(self, mode='human'):
        pass
