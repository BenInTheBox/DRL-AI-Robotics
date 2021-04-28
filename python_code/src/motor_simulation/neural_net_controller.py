import torch

from torch import relu
from numpy import abs, sign
from ..constants import DT

ERROR_SCALING: float = 1. / 500.
D_ERROR_SCALING: float = DT / 100.
INTEGRAL_ERROR_SCALING: float = 1. / 5.


class MotorController(torch.nn.Module):

    def __init__(self):
        super(MotorController, self).__init__()

    def step(self, error: float, d_error: float, integral_error: float) -> float:
        error = error * ERROR_SCALING
        d_error = d_error * D_ERROR_SCALING
        integral_error = integral_error * INTEGRAL_ERROR_SCALING

        x = torch.tensor([[
            error,
            error ** 2,
            sign(error) * abs(error) ** 0.5,
            d_error,
            d_error ** 2,
            sign(d_error) * abs(d_error) ** 0.5,
            integral_error,
            integral_error ** 2,
            sign(integral_error) * abs(integral_error) ** 0.5
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
