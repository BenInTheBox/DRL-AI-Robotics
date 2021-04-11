import numpy as np
import torch

from torch import relu, tanh
from torch.autograd import Variable
from sklearn.metrics import r2_score


class Ext(torch.nn.Module):
    def __init__(self):
        super(Ext, self).__init__()

    def step(self, u: float, s: float) -> float:
        x = torch.tensor([[u, s]])
        return self.forward(x)[0]

    def recurcive_predict(self, s_0: float, u: np.ndarray) -> np.ndarray:
        s = np.zeros_like(u)
        s[0] = s_0
        for i in range(1, len(u)):
            s[i] = self.step(u[i - 1], s[i - 1] / 100.)
        return s


class PlateauNet1Hidden(Ext):
    def __init__(self, n_inputs: int, n_hidden_1: int, n_output: int):
        super(PlateauNet1Hidden, self).__init__()

        self.hidden_1 = torch.nn.Linear(n_inputs, n_hidden_1)
        self.predict = torch.nn.Linear(n_hidden_1, n_output)

    def forward(self, x):
        x = relu(self.hidden_1(x))
        x = self.predict(x)
        return x


class PlateauNet2Hidden(Ext):
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


def train_model(model: torch.nn.Module, x: Variable, y: Variable, n_epoch: int = 200) -> torch.nn.Module:
    optimizer = torch.optim.SGD(model.parameters(), lr=0.00001, momentum=0.5)
    loss_func = torch.nn.MSELoss()

    for t in range(n_epoch):
        y_hat = model(x)
        loss = loss_func(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(loss, r2_score(y_hat.detach().numpy(), y.detach().numpy()))

    return model
