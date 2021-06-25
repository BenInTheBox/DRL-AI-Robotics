from typing import Callable

import pygad.torchga
import pygad
import numpy as np

from pygad import torchga
from ..ball_balancer import BBEnv
from ..DDPG import GeneticController
from ..constants import MAX_ANGLE


def fitness_fn_generator_blackbox(actor: GeneticController, reward_fn: Callable, reward_weight: float) -> Callable:
    global torch_ga
    env = BBEnv(reward_fn, reward_weight)
    env.genetic_generation_reset()

    def fitness_func(solution, sol_idx) -> float:
        model_weights_dict = torchga.model_weights_as_dict(model=actor, weights_vector=solution)
        actor.load_state_dict(model_weights_dict)

        cummulative_reward = 0.
        done: bool = False
        observation: np.ndarray = env.genetic_reset()
        while not done:
            action = actor.act(observation)
            obs, reward, done, _ = env.step(action)
            cummulative_reward += reward

        return cummulative_reward

    return fitness_func


def callback_generation(ga_instance):
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))


def train_ball_controller_genetic(hidden_size: int, reward_fn: Callable, reward_weight: float,
                                  nb_generation: int, population: int, num_parents_mating: int,
                                  parent_selection_type: str, crossover_type: str, mutation_type: str,
                                  mutation_percent_genes: int, keep_parents: int):

    actor = GeneticController(6, hidden_size, 2)
    torch_ga = torchga.TorchGA(model=actor,
                               num_solutions=population)

    initial_population = torch_ga.population_weights

    ga_instance = pygad.GA(num_generations=nb_generation,
                           num_parents_mating=num_parents_mating,
                           initial_population=initial_population,
                           fitness_func=fitness_fn_generator_blackbox(actor, reward_fn, reward_weight),
                           parent_selection_type=parent_selection_type,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           mutation_percent_genes=mutation_percent_genes,
                           keep_parents=keep_parents,
                           on_generation=callback_generation,
                           init_range_low=-2.,
                           init_range_high=2.,
                           gene_space={'low': -2., 'high': 2.},
                           random_mutation_min_val=-0.1,
                           random_mutation_max_val=0.1,
                           allow_duplicate_genes=False)

    ga_instance.run()

    ga_instance.plot_result(title="Iteration vs. Fitness", linewidth=4)

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

    best_solution_weights = torchga.model_weights_as_dict(model=actor,
                                                          weights_vector=solution)
    actor.load_state_dict(best_solution_weights)

    return actor
