import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class JigsawDataset(Dataset):
    def __init__(self, data_dir, csv_file, transform=None, is_train=True):
        self.data_dir = data_dir
        self.transform = transform
        self.is_train = is_train
        
        self.df = pd.read_csv(csv_file)
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['image']
        img_path = os.path.join(self.data_dir, img_name)
        
        image = Image.open(img_path).convert('RGB')
        
        # Split image into 3x3 grid (9 tiles)
        w, h = image.size
        tile_w, tile_h = w // 3, h // 3
        
        tiles = []
        for i in range(3):
            for j in range(3):
                # crop(left, upper, right, lower)
                left = j * tile_w
                upper = i * tile_h
                right = left + tile_w
                lower = upper + tile_h
                tile = image.crop((left, upper, right, lower))
                
                if self.transform:
                    tile = self.transform(tile)
                tiles.append(tile)
                
        tiles_tensor = torch.stack(tiles) # (9, C, H, W)
        
        if self.is_train and 'label' in row:
            label_str = row['label']
            labels = [int(x) for x in str(label_str).strip().split()]
            labels_tensor = torch.tensor(labels, dtype=torch.long)
            return tiles_tensor, labels_tensor
        else:
            puzzle_id = str(row['image']).split('.')[0]
            return tiles_tensor, puzzle_id
