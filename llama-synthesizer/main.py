import os
import yaml
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM, pipeline


def load_config(config_path='/config/config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def load_data(initial_captions_path, neighbors_path):
    initial_captions = pd.read_csv(initial_captions_path, delimiter='|', names=['ID', 'Initial_Caption'])
    neighbors = pd.read_csv(neighbors_path, delimiter='|', header=None, names=['test_image', 'neighbors'])
    print ('Len of initial_captions:', len(initial_captions))
    print ('Len of neighbors:', len(neighbors))
    return initial_captions, neighbors


def get_image_path(image_id, image_dir):
    return Image.open(os.path.join(image_dir, image_id))


def load_model(model_id, access_token):
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token)
    print ('Tokenizer loaded')
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True, 
        bnb_4bit_quant_type="nf4", 
        bnb_4bit_compute_dtype=torch.float16
    )
    print ('Quantization config done')
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
        quantization_config=quantization_config,
    )
    print ('Model loaded')
    
    text_gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    print ('Pipeline created')
    
    return text_gen_pipeline, tokenizer


def parse_generated_text(generated_text, prompt):
    clean_text = generated_text[len(prompt):].strip()
    if clean_text.startswith("assistant\n\n"):
        clean_text = clean_text[len("assistant\n\n"):].strip()
    return clean_text


def generate_captions(initial_captions, neighbors, pipeline, tokenizer, image_dir, batch_size, config):
    results = []
    print ('Batch size:', config['test']['batch_size'])
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        raise ValueError("The tokenizer does not have an eos_token_id. Please check the tokenizer configuration.")

    for i in tqdm(range(0, len(neighbors), batch_size)):
        batch = neighbors[i:i+batch_size]
        batch_images = batch['test_image'].values
        batch_neighbors = batch['neighbors'].values

        for image, neighbor in zip(batch_images, batch_neighbors):
            image_path = get_image_path(image, image_dir)
            messages = [
                {"role": "system", "content": "You are an experienced medical professional assistant."},
                {"role": "image", "content": image_path},
                {"role": "user", "content": initial_captions[initial_captions['ID'] == image]['Initial_Caption'].values[0]},
            ]
            # If there is a neighbor, include it in the prompt
            if not pd.isna(neighbor):
                messages.append({
                    "role": "user",
                    "content": f"In one concise sentence (up to 50 words), describe the content of the medical image {image_path}, "
                               f"specifying the imaging modality (e.g., X-ray, CT, ultrasonography, MRI etc), the organ or body part involved, "
                               f"and any significant findings or abnormalities. Additionally, incorporate relevant details from the neighboring "
                               f"caption: {neighbor} to refine and enhance the draft caption: "
                               f"{initial_captions[initial_captions['ID'] == image]['Initial_Caption'].values[0]} "
                               f"Do not include any additional information that is not present in the image or the neighboring caption."
                })
            else:
                # If there is no neighbor, use a generic prompt
                messages.append({
                    "role": "user",
                    "content": "In one concise sentence (up to 50 words), describe the content of the medical image, "
                               "specifying the imaging modality (e.g., X-ray, CT, ultrasonography, MRI etc.), the organ or body part involved, "
                               "and any significant findings or abnormalities. Do not include any additional information that is not present in the image."
                })

            # Apply the chat template to the messages prompt
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Generate the caption
            outputs = pipeline(
                prompt,
                max_new_tokens= config['test']['max_new_tokens'],
                eos_token_id=[eos_token_id],
                do_sample=False, # if False, beam search is used. if True, sampling is used
                num_beams=config['test']['num_beams'],
            )

            generated_text = outputs[0]["generated_text"]
            # Clean the generated text since the model returns the prompt as well
            cleaned_caption = parse_generated_text(generated_text, prompt)

            results.append({
                'ID': image,
                'Generated_Caption': cleaned_caption
            })
    
    return results


def save_results(results, output_path):
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, sep='|', header=False, index=False)
    print (f'Results saved in {output_path}')


def main():

    config = load_config()

    os.environ["CUDA_VISIBLE_DEVICES"] = config['cuda_device']

    initial_captions, neighbors = load_data(config['initial_captions'], config['neighbors'])

    text_gen_pipeline, tokenizer = load_model(config['MODEL_ID'], config['ACCESS_TOKEN'])

    results = generate_captions(
        initial_captions,
        neighbors,
        text_gen_pipeline,
        tokenizer,
        config['IMAGE_DIR'],
        config['test']['batch_size'],
        config
    )

    save_results(results, config['output_path'])
    print(f"Inference complete. Results saved to '{config['output_path']}'.")


if __name__ == "__main__":
    main()
