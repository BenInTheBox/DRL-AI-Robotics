import torch
import numpy as np
import matplotlib.pyplot as plt

from src import BBEnv, BBEnvPid, trajectory_gen, DT, PidActorCritic

env_test = BBEnvPid()
ac = PidActorCritic(env_test.observation_space, env_test.action_space)
print(ac)

env = BBEnv()
e = np.array([-0.01, - 0.01])
d_e = np.array([0.001, 0.001])
print(env.reward(e, d_e))

l = int(10/DT)
t = trajectory_gen(l)

ft_x = np.abs(np.fft.rfft(t[0, :]))
ft_y = np.abs(np.fft.rfft(t[1, :]))
freq = np.fft.rfftfreq(l) / DT


fig, ax = plt.subplots()
ax.plot(t[0, :])
ax.plot(t[1, :])

ax.set(xlabel='iter', ylabel='pos',
       title='')
ax.grid()

plt.show()

plt.figure(figsize=(10, 7))
plt.fill_between(freq[2:50], ft_x[2:50])
plt.fill_between(freq[2:50], ft_y[2:50])

plt.xlabel('frequency (1/s)')
plt.show()