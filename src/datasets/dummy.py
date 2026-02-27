import torch
from torch.utils.data import Dataset, DataLoader
from src.registry import register_dataset

@register_dataset("DUMMY")
class DummyDataset(Dataset):
    def __init__(self, num_samples=1000, img_size=(3, 224, 224), num_classes=10):
        self.num_samples = num_samples
        self.img_size = img_size
        self.num_classes = num_classes

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        img = torch.randn(*self.img_size)
        label = torch.randint(0, self.num_classes, (1,)).item()
        return {"image": img, "label": label}

    @classmethod
    def get_dataloaders(cls, cfg):
        data_cfg = cfg.get('data', {})
        train_samples = data_cfg.get('train_samples', 1000)
        val_samples = data_cfg.get('val_samples', 200)
        img_size = data_cfg.get('img_size', (3, 224, 224))
        num_classes = data_cfg.get('num_classes', 10)
        
        train_ds = cls(num_samples=train_samples, img_size=img_size, num_classes=num_classes)
        val_ds = cls(num_samples=val_samples, img_size=img_size, num_classes=num_classes)

        train_batch = cfg.get('training', {}).get('batch_size_train', 32)
        val_batch = cfg.get('training', {}).get('batch_size_val', 32)
        num_workers = cfg.get('training', {}).get('num_workers', 4)

        train_loader = DataLoader(
            train_ds, 
            batch_size=train_batch, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_ds, 
            batch_size=val_batch, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True
        )

        return train_loader, val_loader