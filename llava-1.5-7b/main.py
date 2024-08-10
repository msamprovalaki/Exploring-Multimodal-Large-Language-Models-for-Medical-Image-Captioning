# main.py
from init import config, processor, model, train_df, valid_df, dev_df, output_dir
from dataset import LlavaDataset
from lightning_module import LlavaModelPLModule
from model import setup_model_for_training
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
        texts.append(gold_caption)
    
    pixel_values = processor(images=images, return_tensors="pt").pixel_values
    input_ids = processor(text=texts, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt").input_ids

    labels = input_ids.masked_fill(input_ids == processor.tokenizer.pad_token_id, -100)

    return input_ids, pixel_values, labels

def eval_collate_fn(examples, max_length=config['hyperparameters']['max_length']):
    images = []
    texts = []
    
    for example in examples:
        image, gold_caption = example
        images.append(image)
        texts.append(gold_caption)

    pixel_values = processor(images=images, return_tensors="pt").pixel_values
    input_ids = processor(text=texts, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt").input_ids

    return input_ids, pixel_values, texts

model = setup_model_for_training(model, config)

# Callbacks
checkpoint_callback = L.callbacks.ModelCheckpoint(
    monitor='val_edit_distance',
    dirpath=output_dir,
    filename='{epoch:02d}-{val_edit_distance:.2f}',
    save_top_k=3,
    mode='min'
)

early_stop_callback = EarlyStopping(monitor='val_edit_distance', patience=3, mode='min')

trainer = L.Trainer(
    accelerator="gpu",
    devices=1,
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
