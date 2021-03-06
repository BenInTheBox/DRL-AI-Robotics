import numpy as np
import torch

from ..constants import BALL_ERROR_SCALING, BALL_D_ERROR_SCALING, BALL_INTEGRAL_ERROR_SCALING, MAX_ANGLE
from spinup.algos.pytorch.ddpg.core import mlp


class BallController(torch.nn.Module):

    def __init__(self):
        super(BallController, self).__init__()

    def step(self, error: float, d_error: float, integral_error: float) -> float:
        error = error * BALL_ERROR_SCALING
        d_error = d_error * BALL_D_ERROR_SCALING
        integral_error = integral_error * BALL_INTEGRAL_ERROR_SCALING

        x = torch.tensor([[
            error,
            d_error,
            integral_error,
        ]])
        return self.forward(x)[0].item()


class PidController(BallController):
    def __init__(self, n_inputs: int, n_output: int):
        super(PidController, self).__init__()

        self.predict = torch.nn.Linear(n_inputs, n_output, bias=False)
        for p in self.parameters():
            p.data.fill_(-0.)

    def forward(self, x):
        x = MAX_ANGLE * torch.tanh(self.predict(x))
        return x


class GeneticController(torch.nn.Module):
    def __init__(self, n_inputs: int, n_hidden: int, n_output: int):
        super(GeneticController, self).__init__()
        self.h1 = torch.nn.Linear(n_inputs, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output, bias=False)
        #for p in self.parameters():
            #p.data.fill_(-0.)

    def act(self, observation: np.ndarray) -> float:
        x = torch.as_tensor(observation, dtype=torch.float32)
        return self.forward(x)[0].detach().numpy()

    def forward(self, x):
        x = torch.tanh(self.h1(x))
        x = torch.tanh(self.predict(x))
        return x


class BlackBoxActorCritic(torch.nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256),
                 activation=torch.nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = BlackBoxActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q = BlackBoxQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).numpy()


def mlp_no_output_bias(sizes, activation, output_activation=torch.nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        if j < len(sizes) - 2:
            act = activation
            layers += [torch.nn.Linear(sizes[j], sizes[j + 1]), act()]
        else:
            act = output_activation
            layers += [torch.nn.Linear(sizes[j], sizes[j + 1], bias=False), act()]
    return torch.nn.Sequential(*layers)


class BlackBoxActor(torch.nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp_no_output_bias(pi_sizes, activation, torch.nn.Tanh)
        self.act_limit = act_limit

    def forward(self, obs):
        # Return output from network scaled to action space limits.
        return self.act_limit * self.pi(obs)

    def act(self, obs):
        # Return output from network scaled to action space limits.
        return self.act_limit * self.pi(obs)


class BlackBoxQFunction(torch.nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.


class PidActorCritic(torch.nn.Module):

    def __init__(self, observation_space, action_space, hidden_sizes=(256, 256),
                 activation=torch.nn.ReLU):
        super().__init__()

        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]

        # build policy and value functions
        self.pi = PidActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        self.q = PidQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs):
        with torch.no_grad():
            return self.pi(obs).numpy()


class PidActor(torch.nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        pi_sizes = [obs_dim] + list(hidden_sizes) + [act_dim]
        self.pi = mlp(pi_sizes, activation, torch.nn.Sigmoid)
        self.act_limit = act_limit

    def forward(self, obs):
        # Return output from network scaled to action space limits.
        return self.act_limit * self.pi(obs)


class PidQFunction(torch.nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs, act):
        q = self.q(torch.cat([obs, act], dim=-1))
        return torch.squeeze(q, -1)  # Critical to ensure q has right shape.
