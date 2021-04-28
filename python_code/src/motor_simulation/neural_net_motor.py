import numpy as np
import torch
import json

from torch import relu
from torch.autograd import Variable
from sklearn.metrics import r2_score
from ..constants import MOTOR_SPEED_SCALING


class Plateau(torch.nn.Module):
    def __init__(self):
        super(Plateau, self).__init__()

    def step(self, u: float, s: float) -> float:
        x = torch.tensor([[u, s / MOTOR_SPEED_SCALING]])
        return self.forward(x)[0]

    def recurcive_predict(self, s_0: float, u: np.ndarray) -> np.ndarray:
        s = np.zeros_like(u)
        s[0] = s_0
        for i in range(1, len(u)):
            s[i] = self.step(u[i - 1], s[i - 1] / MOTOR_SPEED_SCALING)
        return s


class PlateauPhy(Plateau):

    def __init__(self):
        super(PlateauPhy, self).__init__()
        with open('src/data/motor.json') as json_file:
            data = json.load(json_file)
            print(data)
        self.speed_scaling: float = data['speed_scaling']
        self.u_coef: float = data['phy_i_coef_u']
        self.s_coef: float = data['phy_i_coef_s']

    def step(self, u: float, s: float) -> float:
        return self.u_coef * u + self.s_coef / self.speed_scaling * s


class PlateauNet1Hidden(Plateau):
    def __init__(self, n_inputs: int, n_hidden_1: int, n_output: int):
        super(PlateauNet1Hidden, self).__init__()

        self.hidden_1 = torch.nn.Linear(n_inputs, n_hidden_1)
        self.predict = torch.nn.Linear(n_hidden_1, n_output)

    def forward(self, x):
        x = relu(self.hidden_1(x))
        x = self.predict(x)
        return x * 10


class PlateauNet2Hidden(Plateau):
    def __init__(self, n_inputs: int, n_hidden_1: int, n_hidden_2: int, n_output: int):
        super(PlateauNet2Hidden, self).__init__()

        self.hidden_1 = torch.nn.Linear(n_inputs, n_hidden_1)
        self.hidden_2 = torch.nn.Linear(n_hidden_1, n_hidden_2)
        self.predict = torch.nn.Linear(n_hidden_2, n_output)

    def forward(self, x):
        x = relu(self.hidden_1(x))
        x = relu(self.hidden_2(x))
        x = self.predict(x)
        return x * 10


def train_plateau_model(model: torch.nn.Module, x: Variable, y: Variable, n_epoch: int = 200) -> torch.nn.Module:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.000001, momentum=0.5)
    loss_func = torch.nn.MSELoss()

    for t in range(n_epoch):
        y_hat = model(x)
        loss = loss_func(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(loss, r2_score(y_hat.detach().numpy(), y.detach().numpy()))

    return model
