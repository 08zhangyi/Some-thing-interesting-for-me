from torch.utils.data import Dataset,DataLoader
from glob import glob
import numpy as np
from PIL import Image


class DogsAndCatsDataset(Dataset):
    def __init__(self, root_dir, size=(224, 224)):
        self.files = glob(root_dir)
        self.size = size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = np.asarray(Image.open(self.files[idx]).resize(self.size))
        label = self.files[idx].split('/')[-2]
        return img, label


dogsdset = DogsAndCatsDataset('')
dataloader = DataLoader(dogsdset, batch_size=32, num_workers=2)
for imgs, labels in dataloader:
    pass
