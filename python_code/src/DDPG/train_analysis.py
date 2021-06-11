import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def analyse_training(controller_type='black_box_controller', reward_type='e_r'):
    controllers = os.listdir('src/data/{}/{}'.format(controller_type, reward_type))
    logs = [pd.read_csv('src/data/{}/{}/{}/progress.txt'.format(controller_type, reward_type, controller), sep='\t') for
            controller in controllers]

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 10))

    for controller, log in zip(controllers, logs):
        axs[0].plot(log['AverageEpRet'].rolling(window=20).mean(), label=controller, linewidth=1)
        axs[0].set_ylabel('AvergeEpRet')
        axs[0].legend()
        axs[2].set_xlabel('Epoch')

        axs[1].plot(log['LossPi'])
        axs[1].set_ylabel('Actor loss')
        axs[2].set_xlabel('Epoch')

        axs[2].plot(log['LossQ'])
        axs[2].set_ylabel('Critic loss')
        axs[2].set_xlabel('Epoch')

    plt.show()


"""
def test_controllers(target_trajectory: np.ndarray):
    bb_controllers_names_1 = os.listdir('src/data/{}/{}'.format('black_box_controller', 'e_r'))
    bb_controllers_1 = ['src/data/{}/{}/{}/pyt_save/model.pt'.format('black_box_controller', 'e_r', name) for name in
                        bb_controllers_names_1]
    bb_controllers_names_2 = os.listdir('src/data/{}/{}'.format('black_box_controller', 'de_r'))
    bb_controllers_2 = ['src/data/{}/{}/{}/pyt_save/model.pt'.format('black_box_controller', 'de_r', name) for name in
                        bb_controllers_names_2]
    bb_controllers_names = bb_controllers_names_1 + bb_controllers_names_2
    bb_controllers = bb_controllers_1 + bb_controllers_2

    pid_controllers = os.listdir('src/data/{}/{}'.format('dyn_pid_controller', 'e_r'))
    pid_controllers += os.listdir('src/data/{}/{}'.format('dyn_pid_controller', 'de_r'))

    bb_env = BBEnv(linear_e_reward, 1.)
    scores_bb = [
        grade_bb(controller, bb_env, target_trajectory)
        for controller in bb_controllers
    ]
    print(scores_bb)


def grade_bb(model, env, target_trajectory):
    _, _, _, _, grade = env.simulate(model, target_trajectory)
    return grade


def grade_pid(model, env, target_trajectory):
    env.test_reset()
    _, _, _, _, grade, _, _, _ = env.simulate(model, target_trajectory)
    return grade
"""
