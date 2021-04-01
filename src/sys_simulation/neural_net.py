import torch
import torch.nn.functional as F


class PlateauNet0Hidden(torch.nn.Module):
    def __init__(self, n_inputs: int, n_output: int):
        super(PlateauNet0Hidden, self).__init__()
        self.predict = torch.nn.Linear(n_inputs, n_output)

    def forward(self, x):
        x = self.predict(x)
        return x


class PlateauNet1Hidden(torch.nn.Module):
    def __init__(self, n_inputs: int, n_hidden_1: int, n_output: int):
        super(PlateauNet1Hidden, self).__init__()

        self.hidden_1 = torch.nn.Linear(n_inputs, n_hidden_1)
        self.predict = torch.nn.Linear(n_hidden_1, n_output)

    def forward(self, x):
        x = F.relu(self.hidden_1(x))
        x = self.predict(x)
        return x


class PlateauNet2Hidden(torch.nn.Module):
    def __init__(self, n_inputs: int, n_hidden_1: int, n_hidden_2: int, n_output: int):
        super(PlateauNet2Hidden, self).__init__()

        self.hidden_1 = torch.nn.Linear(n_inputs, n_hidden_1)
        self.hidden_2 = torch.nn.Linear(n_hidden_1, n_hidden_2)
        self.predict = torch.nn.Linear(n_hidden_2, n_output)

    def forward(self, x):
        x = F.relu(self.hidden_1(x))
        x = F.relu(self.hidden_2(x))
        x = self.predict(x)
        return x
