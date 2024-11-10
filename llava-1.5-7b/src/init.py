import yaml
import os
import pandas as pd
from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration
import torch

# Function to load the configuration from a YAML file
def load_config(config_path='../config/config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Function to setup the CUDA device for training
def setup_cuda(cuda_visible_devices):
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

# Function to load dataset (train, valid, dev) and append .jpg extension to image IDs
def load_dataset(train_path, valid_path, dev_path):
    train_df = pd.read_csv(train_path)
    valid_df = pd.read_csv(valid_path)
    dev_df = pd.read_csv(dev_path)

    # Ensure .jpg extension is added to the image IDs
    train_df['ID'] = train_df['ID'].apply(lambda x: x + '.jpg')
    valid_df['ID'] = valid_df['ID'].apply(lambda x: x + '.jpg')
    dev_df['ID'] = dev_df['ID'].apply(lambda x: x + '.jpg')

    return train_df, valid_df, dev_df

# Function to load the processor from the model ID
def load_processor(model_id):
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"  # Set padding side for decoder
    return processor

# Function to load the model with or without LoRA/QLoRA
def load_model(config, model_id, use_lora, use_qlora):
    if use_qlora or use_lora:
        if use_qlora:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float32
            )
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            quantization_config=bnb_config,
        )
    else:
        model = LlavaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            _attn_implementation="flash_attention_2",
        )
    return model

# Function to ensure the output directory exists
def ensure_output_dir(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

def main():
    # Load configuration
    config = load_config()

    # Setup CUDA device
    setup_cuda(config['hyperparameters']['cuda_visible_devices'])

    # Load datasets
    train_df, valid_df, dev_df = load_dataset(
        config['paths']['train_dataset_path'], 
        config['paths']['valid_dataset_path'], 
        config['paths']['dev_dataset_path']
    )

    # Load processor
    processor = load_processor(config['hyperparameters']['model_id'])

    # Load the model with or without LoRA/QLoRA
    model = load_model(
        config, 
        config['hyperparameters']['model_id'],
        config['hyperparameters']['use_lora'],
        config['hyperparameters']['use_qlora']
    )

    # Ensure the output directory exists
    ensure_output_dir(config['paths']['output_dir'])

    return train_df, valid_df, dev_df, processor, model

if __name__ == "__main__":
    main()
