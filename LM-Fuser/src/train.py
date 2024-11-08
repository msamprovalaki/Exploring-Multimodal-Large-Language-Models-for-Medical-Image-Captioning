import os
import yaml
import pandas as pd
import numpy as np
import random
import torch
from datasets import DatasetDict, Dataset
from transformers import T5Tokenizer, DataCollatorForSeq2Seq, T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer, get_scheduler
# from transformers import EarlyStoppingCallback
import nltk
import evaluate



class FlanT5:
    def __init__(self, config_path='/config/config.yaml'):
        self.config = self.load_config(config_path)
        self.set_environment_variables()
        self.set_random_seed(self.config.get('seed', 42))  # Set random seed with default 42
        self.train_df, self.valid_df = self.load_datasets()
        self.dataset = self.prepare_datasets()
        self.tokenizer, self.model = self.load_model_and_tokenizer()
        self.metric = evaluate.load("rouge")

    def load_config(self, config_path):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file '{config_path}' not found.")
        
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        # Check if necessary keys are present in the configuration
        required_keys = ['train_inputs_file', 'valid_inputs_file', 'train_targets_file', 'valid_targets_file', 
                         'model_name', 'max_length', 'output_dir', 'learning_rate', 'batch_size', 'per_device_eval_batch', 
                         'num_epochs', 'seed']  # Added 'seed' to required keys
        for key in required_keys:
            if key not in config:
                raise KeyError(f"Missing required config key: {key}")

        return config
    

    def set_environment_variables(self):
        if 'CUDA_VISIBLE_DEVICES' in self.config:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.config['CUDA_VISIBLE_DEVICES']
            print('Cuda device: ' + os.environ["CUDA_VISIBLE_DEVICES"])
        else:
            print("No CUDA device specified in the configuration. Training might be slower on CPU.")

    def set_random_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        print(f"Random seed set to {seed}")

    def load_datasets(self):
        for file_path in [self.config['data']['train_inputs_file'], 
                          self.config['data']['valid_inputs_file'],
                          self.config['data']['train_targets_file'],
                          self.config['data']['valid_targets_file']]:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Dataset file '{file_path}' not found.")
        
        train_inputs_df = pd.read_csv(self.config['data']['train_inputs_file'], sep='|')
        valid_inputs_df = pd.read_csv(self.config['data']['valid_inputs_file'], sep='|')
        train_targets_df = pd.read_csv(self.config['data']['train_targets_file'])
        valid_targets_df = pd.read_csv(self.config['data']['valid_targets_file'])
        print('Datasets loaded.')

        # Rename columns
        train_inputs_df.columns = self.config['data']['input_columns']
        valid_inputs_df.columns = self.config['data']['input_columns']
        print('Columns renamed.')

        # Append '.jpg' to IDs if missing
        train_targets_df['ID'] = train_targets_df['ID'].apply(lambda x: x if x.endswith('.jpg') else x + '.jpg')
        valid_targets_df['ID'] = valid_targets_df['ID'].apply(lambda x: x if x.endswith('.jpg') else x + '.jpg')

        # Merge inputs and targets
        if all(train_inputs_df.columns == valid_inputs_df.columns):
            train_df = train_inputs_df.merge(train_targets_df, on='ID')
            valid_df = valid_inputs_df.merge(valid_targets_df, on='ID')
        else:
            raise ValueError('Columns of inputs and targets do not match. Please check the files.')

        return train_df, valid_df

    def prepare_datasets(self):
        try:
            train_dataset = Dataset.from_pandas(self.train_df)
            valid_dataset = Dataset.from_pandas(self.valid_df)
        except Exception as e:
            raise RuntimeError(f"Error converting dataframes to Hugging Face datasets.")

        return DatasetDict({'train': train_dataset, 'valid': valid_dataset})

    def load_model_and_tokenizer(self):
        MODEL = self.config['model']['name']
        try:
            tokenizer = T5Tokenizer.from_pretrained(MODEL)
            model = T5ForConditionalGeneration.from_pretrained(MODEL)
            model.to('cuda')
            print('Model and tokenizer loaded.')
        except Exception as e:
            raise RuntimeError(f"Error loading model and tokenizer: {e}")

        return tokenizer, model

    def preprocess_function(self, examples):
        MAX_LENGTH = self.config['hyperparameters']['train']['max_length']

        input_text = (
            f"Using the following three descriptions, generate a single, detailed, and coherent caption that captures the essence of all three while being more concise: "
            f"1. {examples['ModelA']} "
            f"2. {examples['ModelB']} "
            f"3. {examples['ModelC']}"
        )
        
        try:
            model_inputs = self.tokenizer(input_text, max_length=MAX_LENGTH, truncation=True)
            labels = self.tokenizer(text_target=examples['caption'], max_length=MAX_LENGTH, truncation=True)
            model_inputs["labels"] = labels["input_ids"]
        except Exception as e:
            raise RuntimeError(f"Error during tokenization: {e}")

        return model_inputs

    def tokenize_datasets(self):
        try:
            tokenized_dataset = self.dataset.map(self.preprocess_function, batched=True)
            print('Dataset tokenized.')
        except Exception as e:
            raise RuntimeError(f"Error tokenizing datasets: {e}")

        return tokenized_dataset

    def compute_metrics(self, eval_preds):
        preds, labels = eval_preds
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
        result = self.metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        print(result)
        return result

    def train(self):
        OUTPUT_DIR = self.config['hyperparameters']['train']['output_dir']
        EPOCHS = self.config['hyperparameters']['train']['num_epochs']
        LEARNING_RATE = self.config['hyperparameters']['train']['learning_rate']
        
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        # Early stopping callback
        # early_stopping_callback = EarlyStoppingCallback(
        #     early_stopping_patience=self.config['hyperparameters']['train']['early_stopping_patience'],  
        #     early_stopping_threshold = self.config['hyperparameters']['train']['early_stopping_threshold']
        # )

        # Set up the training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=OUTPUT_DIR,
            evaluation_strategy="steps",
            learning_rate=LEARNING_RATE,
            per_device_train_batch_size=self.config['hyperparameters']['train']['batch_size'],
            per_device_eval_batch_size=self.config['hyperparameters']['eval']['per_device_eval_batch'],
            num_train_epochs=EPOCHS,
            predict_with_generate=True,
            save_strategy="no", # yes to save checkpoints
            logging_dir=f"{OUTPUT_DIR}/logs",
            logging_steps=500,  # Log every 500 steps (adjust as needed)
            save_steps=1000,  # Save model checkpoints every 1000 steps (optional)
            # Uncomment to use early stopping
            # early_stopping_patience=self.config['hyperparameters']['train']['early_stopping_patience'],
            # load_best_model_at_end=True,  
            # metric_for_best_model="rougeL",  
            # greater_is_better=True  
        )

        # Initialize the Trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.tokenized_dataset["train"],
            eval_dataset=self.tokenized_dataset["valid"],
            tokenizer=self.tokenizer,
            data_collator=DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model),
            compute_metrics=self.compute_metrics,
            # Uncomment to use early stopping
            # callbacks=[early_stopping_callback]
        )

        # Implement learning rate scheduler (to decrease LR each epoch)
        lr_scheduler = get_scheduler(
            name="linear",  
            optimizer=trainer.optimizer,
            num_warmup_steps=0, 
            num_training_steps=trainer.args.max_steps,  
        )
        trainer.create_optimizer_and_scheduler(num_training_steps=trainer.args.max_steps)
        trainer.lr_scheduler = lr_scheduler  

        # Training loop
        trainer.train()

        # Save the model and tokenizer
        self.model.save_pretrained(OUTPUT_DIR)
        self.tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"Model and tokenizer saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    try:
        flanT5 = FlanT5()
        flanT5.tokenized_dataset = flanT5.tokenize_datasets()
        flanT5.train()
        print('Training completed.')
    except Exception as e:
        print(f"An error occurred: {e}")
