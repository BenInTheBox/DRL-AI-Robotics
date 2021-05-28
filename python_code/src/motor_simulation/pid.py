import json
import numpy as np


class Pid:

    def __init__(self):
        with open('src/data/motor_pid.json') as json_file:
            data = json.load(json_file)
            print(data)
        self.w: np.ndarray = np.array(data['weights'])
        self.b: float = data['bias']

    def step(self, error: np.ndarray) -> float:
        return np.dot(self.w, error) + self.b
