import json
import numpy as np

from ..constants import MAX_MOTOR_U


class Pid:

    def __init__(self):
        with open('src/data/motor_pid.json') as json_file:
            data = json.load(json_file)
            print(data)
        self.w: np.ndarray = np.array(data['weights'])

    def step(self, error: np.ndarray) -> float:
        return np.tanh(np.dot(self.w, error)) * MAX_MOTOR_U

