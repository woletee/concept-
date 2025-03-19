from flask import Flask, request, jsonify
from Vit_concept import run_inference, model
from GP import genetic_programming
import traceback
import os

app = Flask(__name__)

@app.route('/')
def home():
    return "API is running."

@app.route('/run', methods=['POST'])
def run_model():
    try:
        data = request.get_json(force=True)  # force=True handles edge cases where headers are weird
        input_output_pairs = []
        predicted_HLCs = []
        
        # Debugging: Log sample count
        print(f"Received {len(data.get('train', []))} training samples")
        
        for sample in data["train"]:
            input_grid = sample["input"]
            output_grid = sample["output"]
            
            # Debug step
            print("Running run_inference on a sample...")
            concept_label, *_ = run_inference(model, input_grid, output_grid)
            predicted_HLCs.append(concept_label)
            input_output_pairs.append((input_grid, output_grid))
            
        predicted_HLCs = list(set(predicted_HLCs))
        
        print("Calling genetic_programming...")
        best_program, generations = genetic_programming(
            input_output_pairs=input_output_pairs,
            population_size=300,
            generations=500,
            mutation_rate=0.2,
            crossover_rate=0.7,
            max_depth=3,
            predicted_HLCs=predicted_HLCs
        )
        
        print("Returning response...")
        return jsonify({
            "best_program": str(best_program)
        })
    except Exception as e:
        print("🔥 ERROR in /run route!")
        print(traceback.format_exc())  # Show full error trace in Render logs
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    print(f"Starting server on port {port}...")
    app.run(host="0.0.0.0", port=port)
