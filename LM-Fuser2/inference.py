import os
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration, LogitsProcessor
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load config from YAML
def load_config(config_path='/config/config.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

# Load tokenizer and model
def load_model_and_tokenizer(model_dir):
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    model.to('cuda')
    return tokenizer, model

# Load the Sentence Transformer for embeddings
def load_embedding_model(embedding_model_name):
    return SentenceTransformer(embedding_model_name)

# Custom Logits Processor to boost similar captions
class SimilarityLogitsProcessor(LogitsProcessor):
    def __init__(self, candidate_captions, embedding_model, boost_factor, tokenizer):
        self.candidate_captions = candidate_captions
        self.embedding_model = embedding_model
        self.boost_factor = boost_factor
        self.tokenizer = tokenizer  

    def __call__(self, input_ids, scores):
        generated_caption = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        captions = [generated_caption] + self.candidate_captions
        embeddings = self.embedding_model.encode(captions)
        similarities = cosine_similarity(embeddings)

        seen_tokens = set()
        for idx, similarity in enumerate(similarities[0][1:]):
            candidate_tokens = self.candidate_captions[idx].split()
            for token in candidate_tokens:
                token_id = self.tokenizer.encode(token, add_special_tokens=False)
                if token_id and token_id[0] not in seen_tokens and token_id[0] < len(scores):
                    scores[token_id[0]] += np.mean(similarities) * self.boost_factor
                    seen_tokens.add(token_id[0])
        return scores

# Function to generate predictions
def generate_predictions(df, tokenizer, model, embedding_model, config):
    predictions = []
    batch_size = config['hyperparameters']['test']['batch_size']
    for i in tqdm(range(0, len(df), batch_size)):
        batch_df = df.iloc[i:i + batch_size]
        batch_input_texts = []

        for _, row in batch_df.iterrows():
            captionA = row['ModelA']
            captionB = row['ModelB']
            captionC = row['ModelC']

            input_text = (
                f"Given the following three descriptions, synthesize a single, detailed, and concise caption that effectively captures the core elements of each: "
                f"1. {captionA} "
                f"2. {captionB} "
                f"3. {captionC} "
                f"The final caption should seamlessly weave together the key points from all three descriptions, maintaining a clear, unified narrative that avoids repetition while highlighting their unique contributions."
            )
            batch_input_texts.append(input_text)

        batch_input_ids = tokenizer(batch_input_texts, return_tensors="pt", padding=True, truncation=True).input_ids.to("cuda")
        
        batch_outputs = model.generate(
            batch_input_ids,
            max_length=config['hyperparameters']['test']['max_length'],
            num_beams=config['hyperparameters']['test']['num_beams'],
            return_dict_in_generate=True,
            output_scores=True,
            logits_processor=[SimilarityLogitsProcessor(
                candidate_captions=[captionA, captionB, captionC],
                embedding_model=embedding_model,
                boost_factor=config['hyperparameters']['test']['boost_factor'],
                tokenizer=tokenizer  
            )]
        )

        for j, output in enumerate(batch_outputs.sequences):
            output_text = tokenizer.decode(output, skip_special_tokens=True)
            predictions.append(output_text.strip())
    return predictions

# Function to save predictions to CSV
def save_predictions(df, predictions, output_path):
    predictions_df = pd.DataFrame({'ID': df['ID'], 'final_prediction': predictions})
    predictions_df.to_csv(output_path, header=None, index=False)
    print(f'Predictions saved to {output_path}')

# Main function
def main():
    # Load config
    config = load_config('/config/config.yaml')

    # Set CUDA device
    os.environ["CUDA_VISIBLE_DEVICES"] = config['CUDA_VISIBLE_DEVICES']

    # Load tokenizer, model, and embedding model
    tokenizer, model = load_model_and_tokenizer(config['model']['name'])
    embedding_model = load_embedding_model(config['model']['embedding_model_name'])

    # Load dataset
    df = pd.read_csv(config['data']['dev_inputs_file'], header=None)
    df.columns = config['hyperparameters']['test']['column_names']

    # Generate predictions
    predictions = generate_predictions(df, tokenizer, model, embedding_model, config)

    # Save predictions to file
    save_predictions(df, predictions, config['data']['output_file'])

if __name__ == "__main__":
    main()
