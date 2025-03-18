import random 
from dsl import *
# Dictionary of DSLs or primitive functions and terminals

import random

# Dictionary of DSLs or primitive functions and terminals
FUNCTIONS_dictionary = {
    
    'reverse_object_top_bottom':(reverse_object_top_bottom,1),
    'flip_horizontal': (flip_horizontal, 1),
    'flip_vertical': (flip_vertical, 1),
    'rotate_90': (rotate_90, 1),
    'rotate_180': (rotate_180, 1),
    'rotate_270': (rotate_270, 1),
    'identity': (identity, 1),
    'transform_blue_to_red': (transform_blue_to_red, 1),
    'vertical_mirror': (vmirrors, 1),
    'horizontal_mirror': (hmirror, 1),
    'diagonal_mirror': (diamirror, 1),
    'find_center_pixel':(find_center_pixel,1),
    'get_object_bounds': (get_object_bounds,1),
    'reverse_object_top_bottom':(reverse_object_top_bottom,1)
    # Uncommented functions (if needed)
    # 'extract_largest_row': (extract_largest_row, 1),
    # 'extract_bottom_object': (extract_bottom_object, 1),
    # 'extract_topmost_object': (extract_topmost_object, 1),
    # 'fill_downward': (fill_downward, 1),
    # 'keep_bottom_object': (keep_bottom_object, 1),
    # 'remove_least_dominant_pixel': (remove_least_dominant_pixel, 1),
    # 'remove_center_object': (remove_center_object, 1),
    # 'remove_below_horizontal_line': (remove_below_horizontal_line, 1),
    # 'swap_objects': (swap_objects, 1),
    # 'draw_horizontal_vertical': (draw_horizontal_vertical, 1),
}

TERMINALS = [
    ('input_grid', lambda: None, 0)  
]

class Node:
    _id_counter = 0
    
    def __init__(self, value, children=None):
        self.id = Node._id_counter
        Node._id_counter += 1
        self.value = value
        self.children = children if children is not None else []

    def __str__(self):
        if self.children:
            return f"{self.value}({', '.join(str(child) for child in self.children)})"
        else:
            return str(self.value)

    def evaluate(self, input_grid):
        if not self.children:
            if self.value == "input_grid":
                return input_grid
            else:
                raise ValueError(f"Unknown terminal: {self.value}")
        else:
            child_values = [child.evaluate(input_grid) for child in self.children]
            func_data = FUNCTIONS_dictionary.get(self.value)
            if func_data is None:
                raise ValueError(f"Unknown function: {self.value}")
            func, _ = func_data
            return func(*child_values)

def generate_random_program(max_depth, current_depth=0):
    if current_depth >= max_depth or (current_depth > 0 and random.random() < 0.2):
        terminal = random.choice(TERMINALS)
        return Node(terminal[0])
    else:
        func_name, (func, arity) = random.choice(list(FUNCTIONS_dictionary.items()))
        children = [generate_random_program(max_depth, current_depth + 1) for _ in range(arity)]
        return Node(func_name, children)

def get_all_nodes(program):
    nodes = [program]
    for child in program.children:
        nodes.extend(get_all_nodes(child))
    return nodes

class Generation:
    def __init__(self, best_fitness, population, mutation_rate, crossover_rate, max_depth):
        self.best_fitness = best_fitness
        self.population = population
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.max_depth = max_depth

    def to_dict(self):
        return {
            "best_fitness": self.best_fitness,
            "population": [str(individual) for individual in self.population]
        }
