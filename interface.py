import streamlit as st
import json
from nn import run_inference, model
from gen import genetic_programming

st.title(" Concept Guieded GP_ARC Solver")
st.write("Upload your ARC task (JSON format) and let the system solve it.")

uploaded_file = st.file_uploader("Upload your ARC task JSON file", type=["json"])

if uploaded_file:
    data = json.load(uploaded_file)

    input_output_pairs = []
    predicted_HLCs = []

    st.write("### Running Recognition Module...")

    for sample in data.get("train", []):  # or 'test'
        input_grid = sample["input"]
        output_grid = sample["output"]

        st.write("#### Input Grid:")
        st.text(input_grid)
        st.write("#### Output Grid:")
        st.text(output_grid)

        concept_label, _ = run_inference(model, input_grid, output_grid)
        st.write(f" Predicted Concept: `{concept_label}`")

        predicted_HLCs.append(concept_label)
        input_output_pairs.append((input_grid, output_grid))

    predicted_HLCs = list(set(predicted_HLCs))
    st.write("### Predicted High-Level Concepts:", predicted_HLCs)

    if st.button("Run Genetic Programming"):
        st.write("Running Genetic Programming... (this may take a few minutes)")
        best_program, generations = genetic_programming(
            input_output_pairs=input_output_pairs,
            population_size=300,
            generations=500,
            mutation_rate=0.2,
            crossover_rate=0.7,
            max_depth=3,
            predicted_HLCs=predicted_HLCs
        )
        st.success(" GP Completed!")
        st.write("### Best Program Found:")
        st.code(str(best_program))
