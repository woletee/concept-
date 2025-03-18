from flask import Flask, request, jsonify
from Vit_concept import run_inference, model
from GP import genetic_programming

app = Flask(__name__)

@app.route('/')
def home():
    return "API is running."

@app.route('/run', methods=['POST'])
def run_model():
    try:
        data = request.get_json()

        input_output_pairs = []
        predicted_HLCs = []

        for sample in data["train"]:
            input_grid = sample["input"]
            output_grid = sample["output"]
            concept_label, _ = run_inference(model, input_grid, output_grid)
            predicted_HLCs.append(concept_label)
            input_output_pairs.append((input_grid, output_grid))

        predicted_HLCs = list(set(predicted_HLCs))

        best_program, generations = genetic_programming(
            input_output_pairs=input_output_pairs,
            population_size=300,
            generations=500,
            mutation_rate=0.2,
            crossover_rate=0.7,
            max_depth=3,
            predicted_HLCs=predicted_HLCs
        )

        return jsonify({
            "best_program": str(best_program),
            "generations": generations
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Use Render's PORT or default to 5000
    app.run(host='0.0.0.0', port=port)
