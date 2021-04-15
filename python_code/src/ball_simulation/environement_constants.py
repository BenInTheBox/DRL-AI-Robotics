import numpy as np

from gym import spaces
from control import TransferFunction, tf


class BBProps:
    """
    Ball balancer physical properties
    """

    def __init__(self):
        """
        Parameters init
        """
        self.plate_length: float = 0.2  # m

        self.ball_inertia: float = 1.  # kg/m²
        self.ball_mass: float = 0.001  # kg
        self.ball_radius: float = 0.002  # m

        self.arm_length: float = 0.05  # m

        self.srv_k: float = 1.53  # rad/(V-s)
        self.srv_t: float = 0.0248  # s

        self.sampling_period: float = 0.0001  # s

        self.k: float = None
        self.tf: TransferFunction = None

        self.observation_space: spaces = spaces.Box(low=-self.plate_length, high=self.plate_length, shape=(1, 2, 1),
                                                    dtype=np.float64)  # [x, y, x_d, y_d, e_x, e_y, d_e_x, d_e_y, i_e_x, i_e_y]

    def show_props(self) -> dict:
        """
        :return: Dict containing the ball balancer properties
        """
        return {"plate_length": self.plate_length, "ball_inertia": self.ball_inertia,
                "ball_mass": self.ball_mass, "ball_radius": self.ball_radius,
                "arm_length": self.arm_length, "srv_k": self.k,
                "srv_t": self.srv_t, "k": self.k, "sampling_period": self.sampling_period}

    def compute_k_tf(self):
        """
        Compute K according to the physical properties
        """
        self.k = 2 * self.ball_mass * 9.81 * self.arm_length * self.ball_radius ** 2 / (
                self.plate_length * (self.ball_mass * self.ball_radius ** 2 + self.ball_inertia))
        self.tf = tf([self.k], [self.srv_t, 1., 0., 0., 0.], )

    def compute_dynamics(self, v: np.ndarray) -> np.ndarray:
        pass
