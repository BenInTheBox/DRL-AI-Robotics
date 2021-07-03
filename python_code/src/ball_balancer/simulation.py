import numpy as np

from ..motor_simulation.simulation import MotorSimulation
from ..motor_simulation.pid import Pid
from ..ball_simulation.simulation import BallSimulation, BallSimulation1D
from ..constants import DT


class BbSimulation:

    def __init__(self):
        # Motors
        self.motor_x: MotorSimulation = MotorSimulation()
        self.motor_y: MotorSimulation = MotorSimulation()
        self.motor_pid_x = Pid()
        self.motor_pid_y = Pid()

        # Ball
        self.ball: BallSimulation = BallSimulation()

        # Motors error
        self.error: np.ndarray = np.array([0., 0.])
        self.d_error: np.ndarray = np.array([0., 0.])
        self.i_error: np.ndarray = np.array([0., 0.])

    def reset_bb(self):
        self.motor_x.reset_motor()
        self.motor_y.reset_motor()

        self.ball.reset_ball()

        self.error: np.ndarray = np.array([0., 0.])
        self.d_error: np.ndarray = np.array([0., 0.])
        self.i_error: np.ndarray = np.array([0., 0.])

    def step_bb(self, target: np.ndarray = np.array([0., 0.])):
        error = target - np.array([self.motor_x.angle, self.motor_y.angle])
        self.d_error = (error - self.error) / DT
        self.i_error += error * DT
        self.error = error

        motor_x_u: float = self.motor_pid_x.step(np.array([self.error[0], self.d_error[0], self.i_error[0]]))
        motor_y_u: float = self.motor_pid_y.step(np.array([self.error[1], self.d_error[1], self.i_error[1]]))

        self.motor_x.step(motor_x_u)
        self.motor_y.step(motor_y_u)
        self.ball.step(self.motor_x.angle, self.motor_y.angle)


class BbSimulation1D:

    def __init__(self):
        # Motors
        self.motor: MotorSimulation = MotorSimulation()
        self.motor_pid = Pid()

        # Ball
        self.ball: BallSimulation1D = BallSimulation1D()

        # Motors error
        self.error: float = 0.
        self.d_error: float = 0.
        self.i_error: float = 0.

    def reset_bb(self):
        self.motor.reset_motor()

        self.ball.reset_ball()

        self.error: float = 0.
        self.d_error: float = 0.
        self.i_error: float = 0.

    def step_bb(self, target: float = 0.):
        error = target - self.motor.angle
        self.d_error = (error - self.error) / DT
        self.i_error += error * DT
        self.error = error

        motor_u: float = self.motor_pid.step(np.array([self.error, self.d_error, self.i_error]))

        self.motor.step(motor_u)
        self.ball.step(self.motor.angle)


def loss(error: np.ndarray) -> float:
    return - np.power(np.apply_along_axis(np.linalg.norm, 0, error), 2.).mean()


"""class BenchmarkEvaluator:

    def __init__(self, target_trajectory: np.ndarray):
        super(BenchmarkEvaluator, self).__init__()
        self.target_trajectory = target_trajectory
        self.dt = DT

        self.phy_model: BbSimulation = BbSimulation()

    def simulate(self, model: BallController) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        self.phy_model.reset_bb()
        trajectory = np.zeros_like(self.target_trajectory)
        error = np.zeros_like(self.target_trajectory)
        u = np.zeros_like(self.target_trajectory)
        angle = np.zeros_like(self.target_trajectory)
        error[:, 0] = self.target_trajectory[:, 0]
        integral: np.ndarray = error[:, 0] * self.dt

        for i in range(1, self.target_trajectory.shape[1]):
            trajectory[:, i] = self.phy_model.ball.x
            error[:, i] = self.phy_model.ball.x - self.target_trajectory[:, i]
            d_error: np.ndarray = (error[:, i] - error[:, i - 1]) / self.dt
            integral += error[:, i] * self.dt

            u[0, i] = model.step(error[0, i], d_error[0], integral[0])
            u[1, i] = model.step(error[1, i], d_error[1], integral[1])

            angle[0, i] = self.phy_model.motor_x.angle
            angle[1, i] = self.phy_model.motor_y.angle

            self.phy_model.step_bb(u[:, i])

        return trajectory, error, u, angle, loss(error)

    def evaluate(self, model: BallController) -> float:
        _, _, _, _, loss = self.simulate(model)
        # print("Loss", loss)
        return loss"""
