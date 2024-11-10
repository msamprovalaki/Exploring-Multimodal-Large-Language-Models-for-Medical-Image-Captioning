import os
import pandas as pd
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration
import yaml

# Load configuration file
def load_config(config_path='/config/config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_config()

# Constants
MODEL_DIR = config['hyperparameters']['test']['model_dir']
INPUT_FILE = config['data']['dev_inputs_file']
OUTPUT_FILE = config['hyperparameters']['test']['output_dir']
MAX_LENGTH = config['hyperparameters']['test']['max_length']
NUM_BEAMS = config['hyperparameters']['test']['num_beams']
BATCH_SIZE = config['hyperparameters']['test']['batch_size']


# Load model and tokenizer
def load_model_and_tokenizer(model_dir):
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    model.to('cuda')
    return tokenizer, model

# Load data
def load_data(input_file):
    df = pd.read_csv(input_file, sep='|', header=None)
    df.columns = config['hyperparameters']['test']['column_names']
    return df

# Generate predictions
def generate_predictions(df, tokenizer, model):
    predictions = []

    for i, row in tqdm(df.iterrows(), total=len(df), batch_size=BATCH_SIZE):
        captionA = row['Model A']
        captionB = row['Model B']
        captionC = row['Model C']

        input_text = (
            f"Using the following three descriptions, generate a single, detailed, and coherent caption that captures the essence of all three while being more concise: "
            f"1. {captionA} "
            f"2. {captionB} "
            f"3. {captionC} "
            f"Ensure that the final caption seamlessly integrates key elements from each description and maintains a clear and unified narrative."
        )

        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
        output = model.generate(input_ids, max_length=MAX_LENGTH, num_beams=NUM_BEAMS, early_stopping=True)
        output_text = tokenizer.decode(output[0], skip_special_tokens=True)
        predictions.append(output_text)

    return predictions

# Save predictions
def save_predictions(predictions, df, output_file):
    predictions_df = pd.DataFrame({'ID': df['ID'], 'final_prediction': predictions})
    predictions_df.to_csv(output_file, sep='|', header=None, index=False)
    print(f'Predictions saved to {output_file}')

# Main function to run the process
def main():
    tokenizer, model = load_model_and_tokenizer(MODEL_DIR)
    df = load_data(INPUT_FILE)
    predictions = generate_predictions(df, tokenizer, model)
    save_predictions(predictions, df, OUTPUT_FILE)

if __name__ == "__main__":
    main()
