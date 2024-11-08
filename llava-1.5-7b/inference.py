from transformers import AutoProcessor, BitsAndBytesConfig, LlavaForConditionalGeneration
import torch
import pandas as pd
from datasets import Dataset, DatasetDict
from PIL import Image
import os
from tqdm import tqdm
from init import config


# Load dataset and append '.jpg' extension
def load_dataset(dataset_path, image_dir):
    dev_df = pd.read_csv(dataset_path)
    dev_df['ID'] = dev_df['ID'].apply(lambda x: x + '.jpg')
    dev_dataset = Dataset.from_pandas(dev_df)
    dataset = DatasetDict({'dev': dev_dataset})
    return dataset


# Get image path from the directory
def get_image_path(image_id, image_dir):
    return Image.open(os.path.join(image_dir, image_id))


# Load model and processor
def load_model_and_processor(model_id, model_path, quantization_config):
    processor = AutoProcessor.from_pretrained(model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        quantization_config=quantization_config,
    )
    model.eval()  # Set model to evaluation mode
    return processor, model


# Generate captions for the batch
def generate_captions(batch_examples, model, processor, max_new_tokens, num_beams):
    batch_images = [get_image_path(example["ID"], DIR) for example in batch_examples]
    prompts = [
        (
            "USER: <image>\n"
            "As an experienced medical professional, describe the content of the image along with the organ depicted and any important findings?\n"
            f"ASSISTANT:"
        ) for _ in batch_examples
    ]

    inputs = processor(text=prompts, images=batch_images, return_tensors="pt", padding=True).to("cuda")

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        early_stopping=True
    )

    generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

    return generated_texts


# Extract captions and append to results
def process_results(batch_examples, generated_texts):
    results = []
    for example, generated_text in zip(batch_examples, generated_texts):
        assistant_caption = generated_text.split("ASSISTANT:")[-1].strip()  # Extract response
        print(f"Generated Caption: {assistant_caption}")
        results.append({
            'ID': example["ID"],
            'Generated_Caption': assistant_caption
        })
    return results


# Save results to CSV
def save_results(results, model_name, output_path):
    results_df = pd.DataFrame(results)
    output_path = output_path.format(model_name=model_name)
    results_df.to_csv(output_path, index=False, sep='|', header=False)
    print(f"Inference complete. Results saved to '{output_path}'.")


def main():
    # Load dataset
    dataset = load_dataset(config['paths']['dev_dataset_path'], config['paths']['dir'])

    # Load processor and model
    processor, model = load_model_and_processor(
        config['hyperparameters']['model_id'],
        config['hyperparameters']['model'],
        BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16
        )
    )

    # Set hyperparameters
    batch_size = config['hyperparameters']['test']['batch_size']
    num_beams = config['hyperparameters']['test']['num_beams']
    max_new_tokens = config['hyperparameters']['test']['max_new_tokens']

    results = []

    # Start inference
    print("Starting inference...")
    print(f'Shape of dev dataset: {len(dataset["dev"])}')

    for i in tqdm(range(0, len(dataset["dev"]), batch_size)):
        batch_indices = list(range(i, min(i + batch_size, len(dataset["dev"]))))
        batch_examples = dataset["dev"].select(batch_indices)

        # Generate captions
        generated_texts = generate_captions(batch_examples, model, processor, max_new_tokens, num_beams)

        # Process and store results
        batch_results = process_results(batch_examples, generated_texts)
        results.extend(batch_results)

    # Save the results to a CSV file
    model_name = config['hyperparameters']['model'].split('/')[-1]
    save_results(results, model_name, config['paths']['output_path'])


# Run the main function
if __name__ == "__main__":
    main()
