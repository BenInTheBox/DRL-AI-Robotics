import torch

from torch import relu
from numpy import abs, sign
from ..constants import MOTOR_ERROR_SCALING, MOTOR_D_ERROR_SCALING, MOTOR_INTEGRAL_ERROR_SCALING


class MotorController(torch.nn.Module):

    def __init__(self):
        super(MotorController, self).__init__()

    def step(self, error: float, d_error: float, integral_error: float) -> float:
        error = error * MOTOR_ERROR_SCALING
        d_error = d_error * MOTOR_D_ERROR_SCALING
        integral_error = integral_error * MOTOR_INTEGRAL_ERROR_SCALING

        x = torch.tensor([[
            error,
            d_error,
            integral_error,
        ]])
        return self.forward(x)[0].item()


class PidController(MotorController):
    def __init__(self, n_inputs: int, n_output: int):
        super(PidController, self).__init__()

        self.predict = torch.nn.Linear(n_inputs, n_output)

    def forward(self, x):
        x = self.predict(x)
        return x


class NnController(MotorController):
    def __init__(self, n_inputs: int, n_hidden_1: int, n_output: int):
        super(NnController, self).__init__()

        self.hidden_1 = torch.nn.Linear(n_inputs, n_hidden_1)
        self.predict = torch.nn.Linear(n_hidden_1, n_output)

    def forward(self, x):
        x = relu(self.hidden_1(x))
        x = self.predict(x)
        return x
