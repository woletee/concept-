

import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, T5Config
from torch.nn import CrossEntropyLoss
from custom_t5_vit import CustomT5ForConditionalGeneration
from GP import genetic_programming
from Nods import FUNCTIONS_dictionary
from task_loader import *
TOKENIZER_PATH = r"C:\Users\gebre\OneDrive - GIST\문서\KakaoTalk Downloads\GPARC_concept_with_vit\GPARC\SRC\Model\tokenizer_vs22_extendarctokens"
#MODEL_SAVE_PATH = r"C:\Users\gebre\OneDrive - GIST\문서\KakaoTalk Downloads\GPARC_concept_with_vit\GPARC\SRC\Model\final_cls_model.pt"
MODEL_SAVE_PATH_1=r"C:\Users\gebre\OneDrive - GIST\문서\KakaoTalk Downloads\GPARC_concept_with_vit\GPARC\SRC\Model\final_cls_modell.pt"
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
class CustomT5Config(T5Config):
    def __init__(self, PE_mix_strategy="default", use_objidx="yes",
                 grid_max_height=33, grid_max_width=34, **kwargs):
        super().__init__(**kwargs)
        self.PE_mix_strategy = PE_mix_strategy  
        self.use_objidx = use_objidx
        self.grid_max_height = grid_max_height
        self.grid_max_width = grid_max_width
config = CustomT5Config(
    vocab_size=len(tokenizer),
    d_model=128,
    num_layers=3,
    num_decoder_layers=3,
    num_heads=8,
    d_ff=256,
    dropout_rate=0.1,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id
)
class ConceptDetector(torch.nn.Module):
    def __init__(self, config, num_classes):
        super().__init__()
        self.model = CustomT5ForConditionalGeneration(config)
        self.classifier_head = torch.nn.Linear(config.d_model, num_classes)
        self.loss_fn = CrossEntropyLoss()
    def forward(self, input_ids, attention_mask):
        encoder_outputs = self.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = encoder_outputs.last_hidden_state[:, 0, :]
        logits = self.classifier_head(pooled_output)
        probs = F.softmax(logits, dim=1)
        return probs
def load_model(model_path):
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    num_classes = checkpoint["classifier_head.weight"].shape[0]
    print(f"Detected `num_classes`: {num_classes}")
    model = ConceptDetector(config=config, num_classes=num_classes)
    model.load_state_dict(checkpoint)
    model.eval()
    return model
model = load_model(MODEL_SAVE_PATH_1)
def replace_digits_with_arc(grid):
    return [[f'<arc_{num}>' for num in row] for row in grid]
def pad_2d_list(grid, pad_token='<arc_pad>', target_size=32):
    padded_grid = [row + [pad_token] * (target_size - len(row)) for row in grid]
    while len(padded_grid) < target_size:
        padded_grid.append([pad_token] * target_size)
    return padded_grid
def reformat_arc_tokens(grid):
    padded_tokens_2d = pad_2d_list(grid)
    flattened_tokens = [token for row in padded_tokens_2d for token in row]
    return " ".join(flattened_tokens)
def preprocess_for_inference(input_grid, output_grid):
    input_grid = replace_digits_with_arc(input_grid)
    output_grid = replace_digits_with_arc(output_grid)
    input_tokens = "<s> Input Grid: " + reformat_arc_tokens(input_grid) + " </s>"
    output_tokens = " Output Grid: " + reformat_arc_tokens(output_grid) + " </s>"
    return input_tokens + output_tokens
# Concept Label Mapping
CONCEPT_LABELS =  {'Above_below': 0, 'Below_row_line': 1, 'Center': 2, 'Copy': 3, 'Horizontal_vertical': 4, 'Inside_outside': 5, 'Remove_below_horizontal_line': 6}

CONCEPT_LABELS_INV = {v: k for k, v in CONCEPT_LABELS.items()}

# Map ViT Concept to GP Function
CONCEPT_TO_FUNCTION_MAP = {
    'Center': 'find_center_pixel',
    'Copy': 'identity',
    'Above_below': 'flip_horizontal',
    'color_top_part': 'flip_vertical',
    'Horizontal_vertical':'Horizontal_vertical',
 
}
def run_inference(model, input_grid, output_grid):
    formatted_input = preprocess_for_inference(input_grid, output_grid)
    encoded = tokenizer(formatted_input, return_tensors="pt")
    with torch.no_grad():
        probs = model(encoded["input_ids"], encoded["attention_mask"])
    predicted_class_index = torch.argmax(probs, dim=1).item()
    concept_label = CONCEPT_LABELS_INV.get(predicted_class_index, "Unknown Concept")
    print(f"Predicted class index: {predicted_class_index}")
    print(f"Predicted concept: {concept_label}")
    gp_function_name = CONCEPT_TO_FUNCTION_MAP.get(concept_label, None)
    if gp_function_name is None:
        print(f"Warning: No matching GP function found for concept `{concept_label}`.")
        return concept_label, None
    mapped_function = FUNCTIONS_dictionary.get(gp_function_name, None)

    return concept_label, mapped_function
if __name__ == "__main__":
    # Path to your JSON file
    JSON_DATA_PATH = r"C:\Users\gebre\OneDrive - GIST\문서\KakaoTalk Downloads\GPARC_concept_with_vit\GPARC\SRC\data\AboveBelow3.json"  

    # Load JSON data
    with open(JSON_DATA_PATH, "r") as f:
        data = json.load(f)

    # Loop through both train and test sets
    results = []

    for split_name in ["train", "test"]:
        if split_name in data:
            print(f"\nRunning inference on `{split_name}` set...")
            split_results = []
            for sample in data[split_name]:
                input_grid = sample["input"]
                output_grid = sample["output"]
                predicted_label, mapped_function = run_inference(model, input_grid, output_grid)
                split_results.append({
                    "input": input_grid,
                    "output": output_grid,
                    "predicted_label": predicted_label,
                    "mapped_function": str(mapped_function)  # in case it's a callable
                })
            results.append({
                "split": split_name,
                "predictions": split_results
            })

    # Optionally: save the result to a JSON file
    with open("inference_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nInference completed. Results saved to `inference_results.json`.")



