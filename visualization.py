
import numpy as np
from graphviz import Digraph
import os
import matplotlib.pyplot as plt
import matplotlib.colors as colors

#Note
"visualization functions"


def execute_program(program, input_grid) :
    return program.evaluate(input_grid)

def save_tree_as_dot(program, filename):
    dot = Digraph(comment='Program Tree')
    def add_nodes_edges(node):
        if node.children:
            node_label = f"{node.value}\nID:{node.id}"
            dot.node(str(node.id), node_label, shape='box', style='filled', color='lightblue')
        else:
            node_label = f"{node.value}\nID:{node.id}"
            dot.node(str(node.id), node_label, shape='ellipse', style='filled', color='lightgreen')
        
        for child in node.children:
            dot.edge(str(node.id), str(child.id))
            add_nodes_edges(child)

    add_nodes_edges(program)
    dot.render(filename, view=False, format='png')
    print(f"Program tree saved as {filename}.png")

def generate_composite_dot(all_generations, filename):
    dot = Digraph(comment='All Programs Across Generations', graph_attr={'compound': 'true', 'rankdir': 'TB'})
    dot.attr(rankdir='TB')  
    for gen_idx, generation in enumerate(all_generations):
        with dot.subgraph(name=f'cluster_gen_{gen_idx}') as c:
            c.attr(label=f'Generation {gen_idx}')
            c.attr(style='filled', color='lightgrey')
            c.attr(rank='same')  
            for prog_idx, (program, fitness) in enumerate(generation.programs_with_fitness):
                is_selected = program in generation.selected_programs
                is_best = program == generation.best_program
                if is_best:
                    node_color = 'gold'
                    node_shape = 'doublecircle'
                elif is_selected:
                    node_color = 'orange'
                    node_shape = 'box'
                else:
                    node_color = 'lightblue'
                    node_shape = 'ellipse'
                def add_nodes_edges(node, parent_id: str = None):
                    label = f"{node.value}\nID:{node.id}"
                    dot.node(str(node.id), label, shape=node_shape, style='filled', color=node_color)
                    if parent_id:
                        dot.edge(parent_id, str(node.id))
                    for child in node.children:
                        add_nodes_edges(child, str(node.id))
                add_nodes_edges(program)
    output_path = dot.render(filename, view=False, format='png')
    print(f"All programs across generations saved as {output_path}")

# Visualization Functions
def plot_comparison(input_grid, expected_output, predicted_output, task_number):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(f'Task {task_number}')
    cmap = colors.ListedColormap(['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00', '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.Normalize(vmin=0, vmax=9)

    axs[0].imshow(input_grid, cmap=cmap, norm=norm)
    axs[0].set_title("Input Grid")
    axs[1].imshow(expected_output, cmap=cmap, norm=norm)
    axs[1].set_title("Expected Output")
    axs[2].imshow(predicted_output, cmap=cmap, norm=norm)
    axs[2].set_title("Predicted Output")

    plt.tight_layout()
    plt.show()
    
    
   