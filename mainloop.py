import json
from nn import run_inference, model 
from gen import genetic_programming
JSON_DATA_PATH = r"C:\Users\gebre\OneDrive - GIST\문서\KakaoTalk Downloads\GPARC_concept_with_vit\GPARC\SRC\data\AboveBelow3.json"
with open(JSON_DATA_PATH, "r") as f:
    data = json.load(f)
input_output_pairs = []
predicted_HLCs = []
print("Running Recognition Module...")

for sample in data["train"]:
    input_grid = sample["input"]
    output_grid = sample["output"]

    concept_label, _ = run_inference(model, input_grid, output_grid)
    predicted_HLCs.append(concept_label)
    input_output_pairs.append((input_grid, output_grid))
predicted_HLCs = list(set(predicted_HLCs))
print("Passing Predicted Concepts to Genetic Programming...")
best_program, generations = genetic_programming(
    input_output_pairs=input_output_pairs,

    population_size=300,
    generations=500,
    mutation_rate=0.2,
    crossover_rate=0.7,
    max_depth=3,
    predicted_HLCs=predicted_HLCs
)

print("\nFinal Best Program:\n", best_program)
