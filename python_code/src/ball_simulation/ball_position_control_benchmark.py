from typing import Callable

import pygad.torchga
import pygad
import numpy as np

from pygad import torchga
from ..ball_balancer import BenchmarkEvaluator
from ..DDPG import BallController, PidController


def fitness_fn_generator(model: BallController, target_trajectory: np.ndarray) -> Callable:
    global torch_ga
    evaluator: BenchmarkEvaluator = BenchmarkEvaluator(target_trajectory)

    def fitness_func(solution, sol_idx) -> float:
        model_weights_dict = torchga.model_weights_as_dict(model=model,
                                                           weights_vector=solution)

        model.load_state_dict(model_weights_dict)

        return evaluator.evaluate(model)

    return fitness_func


def callback_generation(ga_instance):
    print("Generation = {generation}".format(generation=ga_instance.generations_completed))
    print("Fitness    = {fitness}".format(fitness=ga_instance.best_solution()[1]))


def train_ball_controller(target_trajectory: np.ndarray, nb_generation: int, population: int) -> PidController:
    model = PidController(3, 1)

    torch_ga = torchga.TorchGA(model=model,
                               num_solutions=population)

    num_parents_mating: int = 5
    initial_population = torch_ga.population_weights
    parent_selection_type: str = "sss"
    crossover_type: str = "single_point"
    mutation_type: str = "random"
    mutation_percent_genes: int = 40
    keep_parents: int = 3

    ga_instance = pygad.GA(num_generations=nb_generation,
                           num_parents_mating=num_parents_mating,
                           initial_population=initial_population,
                           fitness_func=fitness_fn_generator(model, target_trajectory),
                           parent_selection_type=parent_selection_type,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           mutation_percent_genes=mutation_percent_genes,
                           keep_parents=keep_parents,
                           on_generation=callback_generation,
                           init_range_low=0.,
                           init_range_high=2.,
                           gene_space={'low': 0., 'high': 2.},
                           allow_duplicate_genes=False)

    ga_instance.run()

    ga_instance.plot_result(title="Iteration vs. Fitness", linewidth=4)

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

    best_solution_weights = torchga.model_weights_as_dict(model=model,
                                                          weights_vector=solution)
    model.load_state_dict(best_solution_weights)

    return model
