import numpy as np
import torch

from torch import relu
from torch.autograd import Variable
from sklearn.metrics import r2_score
from ..constants import DT


class Plateau(torch.nn.Module):
    def __init__(self):
        super(Plateau, self).__init__()

    def step(self, d_x: float, d_y: float, angle_x: float, angle_y: float) -> float:
        x = torch.tensor([[d_x * 6., d_y * 6., angle_x / 25., angle_y / 25.]])
        return self.forward(x)[0]

    def recurcive_predict(self, x_0: float, y_0: float, d_x_0: float, d_y_0: float, inputs: np.ndarray) -> np.ndarray:
        pos: np.ndarray = np.zeros((inputs.shape[0], 2))
        pos[0, 0] = x_0
        pos[0, 1] = y_0

        d_x = d_x_0
        d_y = d_y_0

        for i in range(1, inputs.shape[0]):
            pred = self.step(d_x, d_y, inputs[i - 1, 0], inputs[i - 1, 1])
            d_x = pred[0].item()
            d_y = pred[1].item()
            print(d_x, d_y)
            pos[i, 0] = pos[i - 1, 0] + d_x * DT
            pos[i, 1] = pos[i - 1, 1] + d_y * DT
        return pos


class BallNet1Hidden(Plateau):
    def __init__(self, n_inputs: int, n_hidden_1: int, n_output: int):
        super(BallNet1Hidden, self).__init__()

        self.hidden_1 = torch.nn.Linear(n_inputs, n_hidden_1)
        self.predict = torch.nn.Linear(n_hidden_1, n_output)

    def forward(self, x):
        x = relu(self.hidden_1(x))
        x = self.predict(x)
        return x * 6.


class BallNet2Hidden(Plateau):
    def __init__(self, n_inputs: int, n_hidden_1: int, n_hidden_2: int, n_output: int):
        super(BallNet2Hidden, self).__init__()

        self.hidden_1 = torch.nn.Linear(n_inputs, n_hidden_1)
        self.hidden_2 = torch.nn.Linear(n_hidden_1, n_hidden_2)
        self.predict = torch.nn.Linear(n_hidden_2, n_output)

    def forward(self, x):
        x = relu(self.hidden_1(x))
        x = relu(self.hidden_2(x))
        x = self.predict(x)
        return x * 6.


def train_ball_model(model: torch.nn.Module, x: Variable, y: Variable, n_epoch: int = 200) -> torch.nn.Module:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.2)
    loss_func = torch.nn.MSELoss()

    for t in range(n_epoch):
        y_hat = model(x)
        loss = loss_func(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(loss, r2_score(y_hat.detach().numpy(), y.detach().numpy()))

    return model