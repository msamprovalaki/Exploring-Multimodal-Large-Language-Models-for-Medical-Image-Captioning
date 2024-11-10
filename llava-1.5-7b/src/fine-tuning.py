# main.py
from init import config, processor, model, train_df, valid_df, dev_df, output_dir
from dataset import LlavaDataset
from lightning_module import LlavaModelPLModule
from torch.utils.data import DataLoader
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.callbacks import Callback
import lightning as L


train_dataset = LlavaDataset(train_df, processor, max_length=config['hyperparameters']['max_length'])
valid_dataset = LlavaDataset(valid_df, processor, max_length=config['hyperparameters']['max_length'])
dev_dataset = LlavaDataset(dev_df, processor, max_length=config['hyperparameters']['max_length'])

def train_collate_fn(examples, max_length=config['hyperparameters']['max_length']):
    images = []
    texts = []
    
    for example in examples:
        image, gold_caption = example
        images.append(image)
        # Construct the text prompt
        prompt = (
            "USER: <image>\n"
            "As a medical expert, please provide a detailed caption for the medical image showing "
            "[describe the content of the image, e.g., 'an MRI scan of the brain' or "
            "'a histopathology slide of lung tissue']. "
            "Include relevant observations, diagnoses if applicable, and any significant findings. "
            "Your caption should be clear and concise, aimed at aiding medical professionals in "
            "understanding the image's clinical significance?\n"
            f"ASSISTANT: {gold_caption}"
        )
        texts.append(prompt)
    
   # Use the processor to encode texts and images into a batch
    batch = processor(text=texts, images=images, padding=True, truncation=True, max_length=max_length, return_tensors="pt")



    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels
    
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    pixel_values = batch["pixel_values"]
    labels = batch["labels"]

    # print(pixel_values.dtype)

    return input_ids, attention_mask, pixel_values, labels

def eval_collate_fn(examples):
    images = []
    texts = []
    answers = []

    for example in examples:
        image, gold_caption = example
        images.append(image)
        prompt = (
            "USER: <image>\n"
            "As a medical expert, please provide a detailed caption for the medical image showing "
            "[describe the content of the image, e.g., 'an MRI scan of the brain' or "
            "'a histopathology slide of lung tissue']. "
            "Include relevant observations, diagnoses if applicable, and any significant findings. "
            "Your caption should be clear and concise, aimed at aiding medical professionals in "
            "understanding the image's clinical significance?\n"
            f"ASSISTANT:"
        )
        
        texts.append(prompt)
        answers.append(gold_caption)



    batch = processor(text=texts, images=images, return_tensors="pt", padding=True)


    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    pixel_values = batch["pixel_values"]

    return input_ids, attention_mask, pixel_values, answers



# Callbacks
checkpoint_callback = L.callbacks.ModelCheckpoint(
    monitor='val_edit_distance',
    dirpath=output_dir,
    filename='{epoch:02d}-{val_edit_distance:.2f}',
    save_top_k=3,
    mode='min'
)

early_stop_callback = EarlyStopping(monitor='val_edit_distance', patience=config['hyperparameters']['early_stopping_patience'], mode='min')

trainer = L.Trainer(
    accelerator="gpu",
    devices=1,
    batch_size=config['hyperparameters']['batch_size'],
    max_epochs=config['hyperparameters']['max_epochs'],
    check_val_every_n_epoch=config['hyperparameters']['check_val_every_n_epoch'],
    gradient_clip_val=config['hyperparameters']['gradient_clip_val'],
    accumulate_grad_batches=config['hyperparameters']['accumulate_grad_batches'],
    precision=16,
    callbacks=[checkpoint_callback, early_stop_callback]
)

lightning_module = LlavaModelPLModule(
    config=config,
    processor=processor,
    model=model,
    train_dataset=train_dataset,
    valid_dataset=valid_dataset,
    train_collate_fn=train_collate_fn,
    eval_collate_fn=eval_collate_fn
)
trainer.fit(lightning_module)
