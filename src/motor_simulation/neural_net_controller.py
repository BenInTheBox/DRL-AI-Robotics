import torch

from torch import relu
from numpy import abs, sign

DT: float = 0.007
ERROR_SCALING: float = 1. / 250.
D_ERROR_SCALING: float = DT / 35.
INTEGRAL_ERROR_SCALING: float = 1. / 10.


class MotorController(torch.nn.Module):
    def __init__(self, n_inputs: int, n_hidden_1: int, n_output: int):
        super(MotorController, self).__init__()

        self.hidden_1 = torch.nn.Linear(n_inputs, n_hidden_1)
        self.predict = torch.nn.Linear(n_hidden_1, n_output)

    def forward(self, x):
        x = relu(self.hidden_1(x))
        x = self.predict(x)
        return x

    def step(self, error: float, d_error: float, integral_error: float) -> float:
        error = error * ERROR_SCALING
        d_error = d_error * D_ERROR_SCALING
        integral_error = integral_error * INTEGRAL_ERROR_SCALING

        x = torch.tensor([[
            error,
            error ** 2,
            sign(error) * abs(error)**0.5,
            d_error,
            d_error ** 2,
            sign(d_error) * abs(d_error)**0.5,
            integral_error,
            integral_error ** 2,
            sign(integral_error) * abs(integral_error)**0.5
        ]])
        return self.forward(x)[0].item()
