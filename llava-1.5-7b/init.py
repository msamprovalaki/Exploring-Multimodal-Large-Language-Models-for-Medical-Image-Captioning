# init.py
import yaml
import os
import pandas as pd
from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration
import torch

def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_config()

os.environ["CUDA_VISIBLE_DEVICES"] = config['hyperparameters']['cuda_visible_devices']

# Load Dataset
train_df = pd.read_csv(config['paths']['train_dataset_path'])
valid_df = pd.read_csv(config['paths']['valid_dataset_path'])
dev_df = pd.read_csv(config['paths']['dev_dataset_path'])

train_df['ID'] = train_df['ID'].apply(lambda x: x + '.jpg')
dev_df['ID'] = dev_df['ID'].apply(lambda x: x + '.jpg')

# Load Processor
processor = AutoProcessor.from_pretrained(config['hyperparameters']['model_id'])
processor.tokenizer.padding_side = "left" # during training, one always uses padding on the left since it is decoder-only

# Load Model
USE_LORA = config['hyperparameters']['use_lora']
USE_QLORA = config['hyperparameters']['use_qlora']
MODEL_ID = config['hyperparameters']['model_id']

if USE_QLORA or USE_LORA:
    if USE_QLORA:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float32
        )
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        quantization_config=bnb_config,
    )
else:
    model = LlavaForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
        _attn_implementation="flash_attention_2",
    )

# Ensure output directory exists
output_dir = config['paths']['output_dir']
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
