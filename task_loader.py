
import matplotlib.pyplot as plt
from matplotlib import colors
import os
import json
import numpy as np

#Notes 

" directory_path:str keeps the path containing the tasks"
"tasks:list of tuples containing the the task filename and the task data "
"task_idx:index of the current task in the current iteration"
"task_file name:str name of the task file being processed"
"task_data: dictionary containing the task data basically the input and the output grids"
"input_output_pairs: list of tuples containing the input and the output grids"

def load_tasks_from_directory(directory_path):
    tasks = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            filepath = os.path.join(directory_path, filename)
            with open(filepath, 'r') as file:
                task_data = json.load(file)
                tasks.append((filename, task_data))
    return tasks

def prepare_input_output_pairs(task_data):
    input_output_pairs = []
    for example in task_data["train"]:
        input_grid = np.array(example["input"], dtype=int)
        output_grid = np.array(example["output"], dtype=int)
        input_output_pairs.append((input_grid, output_grid))
    return input_output_pairs
