import os
import json
import re
import pandas as pd
from tqdm import tqdm
from PIL import Image
from datasets import Dataset, DatasetDict
from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch
import yaml

# Load Configuration
def load_config(config_path="config/config.yaml"):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

# Load Datasets
def load_datasets(train_path, dev_path):
    train_df = pd.read_csv(train_path)
    dev_df = pd.read_csv(dev_path)
    dev_df['ID'] = dev_df['ID'].apply(lambda x: x + '.jpg')
    return DatasetDict({
        'train': Dataset.from_pandas(train_df),
        'dev': Dataset.from_pandas(dev_df[1310:]),
    })

# Neighbors from the training set
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as jsonfile:
        return json.load(jsonfile)

# Load Test Image
def load_image(image_dir, image_id):
    full_image_path = os.path.join(image_dir, image_id)
    return Image.open(full_image_path).convert("RGB")

# Initialize Model and Processor
def initialize_model_and_processor(model_id, device):
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor

# Clean Generated Caption
def clean_generated_caption(caption):
    caption = re.sub(r'\b(\d+\.|\w+\.)\s', '', caption)
    sentences = re.split(r'(?<=\.)\s+', caption)
    seen = set()
    return ' '.join(sentence.strip() for sentence in sentences if not (sentence.strip() in seen or seen.add(sentence.strip())))

# Extract Last Assistant Text
def extract_last_assistant_text(conversation):
    return conversation.split("ASSISTANT:")[-1].strip() if "ASSISTANT:" in conversation else ""

# Process Batch
def process_batch(batch, dataset, processor, model, image_dir, image_captions, device):
    generated_captions = []
    for example in batch:
        image_id = example["ID"]
        raw_image = load_image(image_dir, image_id)
        captions = image_captions.get(image_id, [])
        
        conversation_1 = [{
            "role": "user",
            "content": [
                {"type": "image", "image": raw_image},
                {"type": "text", "text": "Provide a medical caption based on similar examples."}
            ]
        }]
        
        conversation_2 = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": neighbor['caption']},
                    {"type": "image", "image": load_image(image_dir, neighbor['neighbor_image'])}
                ]
            } for neighbor in captions
        ] or [{
            "role": "user",
            "content": [{"type": "text", "text": "Provide a detailed caption for this medical image."}]
        }]
        
        prompts = [
            processor.apply_chat_template(conversation_1, add_generation_prompt=True),
            processor.apply_chat_template(conversation_2, add_generation_prompt=True),
        ]

        inputs = processor(
            images=[raw_image] + [load_image(image_dir, n['neighbor_image']) for n in captions],
            text=prompts,
            return_tensors="pt",
            padding=True
        ).to(device)
        
        generate_ids = model.generate(**inputs, max_new_tokens=100)
        generated_text = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        last_caption = extract_last_assistant_text(generated_text[-1])
        generated_captions.append(clean_generated_caption(last_caption))
    return generated_captions

# Save Generated Captions
def save_generated_captions(captions, dev_df, output_paths):
    captions_df = pd.DataFrame({'ID': dev_df['ID'], 'final_prediction': captions})
    for path in output_paths:
        captions_df.to_csv(path, index=False, sep='|' if path.endswith('.csv') else ',')

# Main Function
def main(config_path="config/config.yaml"):
    config = load_config(config_path)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = config['cuda_device']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = load_datasets(config['paths']['train_dataset'], config['paths']['dev_dataset'])
    image_captions = load_json(config['paths']['json_data'])
    
    model, processor = initialize_model_and_processor(config['model'], device)
    
    generated_captions = []
    for i in tqdm(range(0, len(dataset["dev"]), config['batch_size'])):
        batch = dataset["dev"].select(list(range(i, min(i + config['batch_size'], len(dataset["dev"])))))
        generated_captions += process_batch(batch, dataset, processor, model, config['paths']['image_dir'], image_captions, device)
    
    save_generated_captions(generated_captions, dataset["dev"].to_pandas(), config['output_paths'])

if __name__ == "__main__":
    main()
