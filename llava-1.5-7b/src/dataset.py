import os
from PIL import Image
from torch.utils.data import Dataset
from typing import Any, Dict

class LlavaDataset(Dataset):
    def __init__(self, dataset: str, processor: Any, max_length: int = 300):
        super().__init__()
        self.dataset = dataset
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.dataset[idx]
        image = Image.open(os.path.join(self.dataset['dir'], sample["ID"])).convert("RGB")
        target_sequence = self.dataset['gold_caption'][idx]
        return image, target_sequence
