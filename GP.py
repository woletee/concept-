from collections import deque
from copy import deepcopy

class GPContext:
    def __init__(self, grid):
        self.grid = deepcopy(grid)
        self.objects = []
        self.top_object = None
        self.shift = None


def extract_object(grid):
    ctx = GPContext(grid)
    rows, cols = len(ctx.grid), len(ctx.grid[0])
    visited = [[False]*cols for _ in range(rows)]

    def bfs(r, c, val):
        queue = deque([(r, c)])
        block = []
        visited[r][c] = True
        while queue:
            x, y = queue.popleft()
            block.append((x, y))
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < rows and 0 <= ny < cols and not visited[nx][ny] and ctx.grid[nx][ny] == val:
                    visited[nx][ny] = True
                    queue.append((nx, ny))
        return block

    for r in range(rows):
        for c in range(cols):
            if ctx.grid[r][c] != 0 and not visited[r][c]:
                block = bfs(r, c, ctx.grid[r][c])
                top_row = min(x for x, y in block)
                value = ctx.grid[block[0][0]][block[0][1]]
                ctx.objects.append((top_row, value, block))

    return ctx

def sort_objects_by_column(ctx):
    ctx.objects.sort(key=lambda obj: min(y for x, y in obj[2]))
    return ctx

def sort_object(ctx):
    ctx.objects.sort(key=lambda obj: obj[0])
    return ctx
def move_right_most_object(ctx):
    if not ctx.objects:
        return ctx.grid

    # Get rightmost object: object with largest column index
    rightmost_object = max(ctx.objects, key=lambda obj: max(y for x, y in obj[2]))
    _, value, block = rightmost_object

    for x, y in block:
        ctx.grid[x][y] = 0

    shift = 0
    cols = len(ctx.grid[0])
    while True:
        can_move = True
        for x, y in block:
            new_y = y + shift + 1
            if new_y >= cols or ctx.grid[x][new_y] != 0:
                can_move = False
                break
        if not can_move:
            break
        shift += 1

    for x, y in block:
        ctx.grid[x][y + shift] = value

    return ctx.grid
def move_left_most_object(ctx):
    if not ctx.objects:
        return ctx.grid

    # Get leftmost object: object with smallest column index
    leftmost_object = min(ctx.objects, key=lambda obj: min(y for x, y in obj[2]))
    _, value, block = leftmost_object

    for x, y in block:
        ctx.grid[x][y] = 0

    shift = 0
    while True:
        can_move = True
        for x, y in block:
            new_y = y - (shift + 1)
            if new_y < 0 or ctx.grid[x][new_y] != 0:
                can_move = False
                break
        if not can_move:
            break
        shift += 1

    for x, y in block:
        ctx.grid[x][y - shift] = value

    return ctx.grid

def move_bottom_most_object(ctx):
    if not ctx.objects:
        return ctx.grid

    # Get bottommost object (last one in the list after sorting by top_row)
    bottom_object = ctx.objects[-1]
    _, value, block = bottom_object

    # Remove it from the grid
    for x, y in block:
        ctx.grid[x][y] = 0

    # Compute shift (same as top, move down)
    shift = 0
    rows = len(ctx.grid)
    while True:
        can_move = True
        for x, y in block:
            new_x = x + shift + 1
            if new_x >= rows or ctx.grid[new_x][y] != 0:
                can_move = False
                break
        if not can_move:
            break
        shift += 1

    # Place object
    for x, y in block:
        ctx.grid[x + shift][y] = value

    return ctx.grid

def move_top_most_object(ctx):
    if not ctx.objects:
        return ctx.grid

    # Get topmost object
    top_object = ctx.objects[0]
    _, value, block = top_object

    # Remove it from the grid
    for x, y in block:
        ctx.grid[x][y] = 0

    # Compute shift
    shift = 0
    rows = len(ctx.grid)
    while True:
        can_move = True
        for x, y in block:
            new_x = x + shift + 1
            if new_x >= rows or ctx.grid[new_x][y] != 0:
                can_move = False
                break
        if not can_move:
            break
        shift += 1

    # Place object
    for x, y in block:
        ctx.grid[x + shift][y] = value

    return ctx.grid
def reverse_object_order(ctx):
    ctx.reversed_objects = list(reversed(ctx.objects))
    return ctx

# Step 3: Clear grid and place objects from top downward
def place_objects(ctx):
    rows, cols = len(ctx.grid), len(ctx.grid[0])
    new_grid = [[0]*cols for _ in range(rows)]
    current_row = 0

    for _, value, block in ctx.reversed_objects:
        # Shift block so top of block aligns with current_row
        min_row = min(x for x, y in block)
        row_shift = current_row - min_row

        # Compute height of block to update current_row
        block_height = max(x for x, y in block) - min_row + 1

        for x, y in block:
            new_x = x + row_shift
            if 0 <= new_x < rows:
                new_grid[new_x][y] = value

        current_row += block_height

    return new_grid
from dsl import * 

concept_hierarchy = {
    "Above Below": {
        "remove below horizontal": ["flip_horizontal", "flip_vertical"],
        "Fill below the pattern": ["rotate_90", "rotate_180", "rotate_270"],
        "Move object": ["extract_object", "sort_objects", "move_top_most_object","extract_object""move_bottom_most_object","move_left_most_object","move_right_most_object"],
        "reverse object":[ "place_objects", "reverse_object_order","extract_object"]
    },
    "Clean Up": {
        "Copy the grid content": ["find_center_pixel"]
    }
}

primitive_function_registry = {
    "move_top_most_object": move_top_most_object,
    "move_bottom_most_object": move_bottom_most_object,
    "move_right_most_object":move_right_most_object,
    "move_left_most_object":move_left_most_object,
    "extract_object": extract_object,
    "sort_objects": sort_object,
    "find_center_pixel":find_center_pixel,
    "extract_object": extract_object,
    "reverse_object_order": reverse_object_order,
    "place_objects": place_objects
}

def get_sub_concepts_and_functions(high_level_concepts):
    allowed_functions = []
    for hlc in high_level_concepts:
        sub_concepts = concept_hierarchy.get(hlc, {})
        for slc_funcs in sub_concepts.values():
            allowed_functions.extend(slc_funcs)
    return list(set(allowed_functions))
import random

TERMINALS = ["input_grid"]

class Node:
    def __init__(self, value, children=None):
        self.value = value
        self.children = children if children else []

    def __str__(self):
        if self.children:
            return f"{self.value}({', '.join(str(child) for child in self.children)})"
        return str(self.value)

    def evaluate(self, input_grid):
        if self.value == "input_grid":
            return input_grid
        child_values = [child.evaluate(input_grid) for child in self.children]
        func = primitive_function_registry.get(self.value)
        if func is None:
            raise ValueError(f"Unknown function: {self.value}")
        return func(*child_values)


def generate_random_program(max_depth, current_depth=0, allowed_functions=None):
    if current_depth >= max_depth or (current_depth > 0 and random.random() < 0.2):
        return Node(random.choice(TERMINALS))
    else:
        func_name = random.choice(allowed_functions) if allowed_functions else random.choice(list(primitive_function_registry.keys()))
        children = [generate_random_program(max_depth, current_depth+1, allowed_functions) for _ in range(1)]  # assuming arity=1
        return Node(func_name, children)


def get_all_nodes(program):
    nodes = [program]
    for child in program.children:
        nodes.extend(get_all_nodes(child))
    return nodes


def crossover(parent1, parent2):
    import copy
    child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
    nodes1, nodes2 = get_all_nodes(child1), get_all_nodes(child2)
    point1, point2 = random.choice(nodes1), random.choice(nodes2)
    point1.value, point1.children, point2.value, point2.children = point2.value, point2.children, point1.value, point1.children
    return child1, child2


def mutation(program, max_depth, mutation_rate, allowed_functions):
    import copy
    mutant = copy.deepcopy(program)
    nodes = get_all_nodes(mutant)
    for node in nodes:
        if random.random() < mutation_rate:
            new_subtree = generate_random_program(max_depth=max_depth, current_depth=0, allowed_functions=allowed_functions)
            node.value = new_subtree.value
            node.children = new_subtree.children
    return mutant


def tournament_selection(population, fitness_scores, k):
    selected = []
    for _ in range(len(population)):
        participants = random.sample(list(zip(population, fitness_scores)), k)
        winner = max(participants, key=lambda x: x[1])[0]
        selected.append(winner)
    return selected


class Generation:
    def __init__(self, best_fitness, population):
        self.best_fitness = best_fitness
        self.population = population

    def to_dict(self):
        return {
            "best_fitness": self.best_fitness,
            "population": [str(ind) for ind in self.population]
        }

def evaluate_fitness(program, input_output_pairs):
    score = 0
    for inp, expected in input_output_pairs:
        try:
            result = program.evaluate(inp)
            if result == expected:
                score += 1
        except:
            continue
    return score


def genetic_programming(input_output_pairs, population_size, generations, mutation_rate, crossover_rate, max_depth, predicted_HLCs):
    allowed_functions = get_sub_concepts_and_functions(predicted_HLCs)
    population = [generate_random_program(max_depth, allowed_functions=allowed_functions) for _ in range(population_size)]
    all_generations = []
    best_program = None

    for gen in range(generations):
        fitness_scores = [evaluate_fitness(p, input_output_pairs) for p in population]
        best_fitness = max(fitness_scores)
        best_program = population[fitness_scores.index(best_fitness)]

        selected = tournament_selection(population, fitness_scores, k=3)
        next_generation = []

        while len(next_generation) < population_size:
            if random.random() < crossover_rate and len(selected) >= 2:
                p1, p2 = random.sample(selected, 2)
                c1, c2 = crossover(p1, p2)
                next_generation.extend([c1, c2])
            else:
                parent = random.choice(selected)
                child = mutation(parent, max_depth, mutation_rate, allowed_functions)
                next_generation.append(child)

        population = next_generation[:population_size]
        all_generations.append(Generation(best_fitness, population))
        print(f"Generation {gen} - Best Fitness: {best_fitness}")

    return best_program, all_generations

if __name__ == "__main__":
 if __name__ == "__main__":
    input_output_pairs = [
        (
            [[0,0,0,0,0,0,0,0],
             [0,3,3,3,0,0,0,0],
             [0,0,0,0,0,0,0,0],
             [0,4,4,4,0,0,0,0],
             [0,4,4,4,0,0,0,0],
             [0,4,4,4,0,0,0,0],
             [0,0,0,0,0,3,3,3],
             [0,0,3,3,3,0,0,0]],

            [[0,0,0,0,0,0,0,0],
             [0,0,0,0,0,0,0,0],
             [0,3,3,3,0,0,0,0],
             [0,4,4,4,0,0,0,0],
             [0,4,4,4,0,0,0,0],
             [0,4,4,4,0,0,0,0],
             [0,0,0,0,0,3,3,3],
             [0,0,3,3,3,0,0,0]]
        )
    ]


    predicted_HLCs = ["Above Beloww"]


    best_program, generations = genetic_programming(
        input_output_pairs=input_output_pairs,
        population_size=500,
        generations=700,
        mutation_rate=0.2,
        crossover_rate=0.7,
        max_depth=3,
        predicted_HLCs=predicted_HLCs
    )
    print("\nBest Program:", best_program)    
