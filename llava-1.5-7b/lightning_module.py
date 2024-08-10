# lightning_module.py
import lightning as L
from torch.utils.data import DataLoader
import re
from nltk import edit_distance
import numpy as np
import torch

class LlavaModelPLModule(L.LightningModule):
    def __init__(self, config, processor, model, train_dataset, valid_dataset, train_collate_fn, eval_collate_fn):
        super().__init__()
        self.config = config
        self.processor = processor
        self.model = model        
        self.batch_size = config['hyperparameters']['batch_size']
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.train_collate_fn = train_collate_fn
        self.eval_collate_fn = eval_collate_fn

    def training_step(self, batch, batch_idx):
        input_ids, pixel_values, labels = batch
        outputs = self.model(input_ids=input_ids, 
                             pixel_values=pixel_values, 
                             labels=labels)
        loss = outputs.loss
        self.log("train_loss", loss)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        input_ids, pixel_values, answers = batch
        generated_ids = self.model.generate(input_ids=input_ids, 
                                            pixel_values=pixel_values,
                                            max_new_tokens=300)
        predictions = self.processor.batch_decode(generated_ids[:, input_ids.size(1):], skip_special_tokens=True)
        scores = []
        for pred, answer in zip(predictions, answers):
            pred = re.sub(r"(?:(?<=>) | (?=</s_))", "", pred)
            score = edit_distance(pred, answer) / max(len(pred), len(answer))
            scores.append(score)
        avg_edit_distance = np.mean(scores)
        self.log("val_edit_distance", avg_edit_distance, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return avg_edit_distance
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config['hyperparameters']['lr'])
        return optimizer

    def train_dataloader(self):
        return DataLoader(self.train_dataset, collate_fn=self.train_collate_fn, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, collate_fn=self.eval_collate_fn, batch_size=self.batch_size, shuffle=False, num_workers=4)
