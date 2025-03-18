import numpy as np
from typing import List, Tuple
from Node import Node

def evaluate_fitness(program: Node, input_output_pairs: List[Tuple[np.ndarray, np.ndarray]]) -> int:
    fitness = 0
    for input_grid, expected_output in input_output_pairs:
        try:
            output = program.evaluate(input_grid)
            if np.array_equal(output, expected_output):
                fitness += 1
        except Exception as e:
            print(f"Error during fitness evaluation: {e}")
            pass
    return fitness
