from torch.utils.data import Dataset, DataLoader
from torch import nn
from typing import Dict, List, Optional, Tuple
import re
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as tr
import torch
import cv2
import os
import pandas as pd
import numpy as np
from tqdm.auto import tqdm, trange
from collections import Counter
import glob
from PIL import Image
import random
from torchvision import models


pretrained_transforms = models.ResNet50_Weights.DEFAULT.transforms()
channel_mean = np.array(pretrained_transforms.mean)
channel_std = np.array(pretrained_transforms.std)

image_prepare = tr.Compose([
    tr.ToPILImage(),
    tr.Resize(size=(256, 256)),
    tr.RandomCrop(size=(224, 224)),
    tr.ToTensor(),
    tr.Normalize(mean=channel_mean, std=channel_std),
])
image_prepare_val = tr.Compose([
    tr.ToPILImage(),
    tr.Resize(size=(256, 256)),
    tr.ToTensor(),
    tr.Normalize(mean=channel_mean, std=channel_std),
])

def tokenize(text):
    text = text.lower()
    text = re.sub("[^\w\s]", " ", text)
    text = re.sub("^\s* | \s*$", "", text)
    text = re.split("\s+", text)
    text.insert(0, "<BOS>")
    text.append("<EOS>")
    return text

def get_vocab(unzip_root: str) -> Tuple[Dict[str, int], Dict[int, str]]:
    from_file = pd.read_csv('vocab.tsv', sep='\t')
    tok_to_ind = dict(from_file.values)
    ind_to_tok = dict(from_file[["index", "token"]].values)
    return tok_to_ind, ind_to_tok

tok_to_ind, ind_to_tok = get_vocab(unzip_root=".")

def to_ids(text):
    tokens = tokenize(text)
    tok_to_ind, _ = get_vocab(unzip_root=".")
    return [tok_to_ind.get(token, tok_to_ind['<UNK>']) for token in tokens]

class ImageCaptioningDataset(Dataset):
    def __init__(self, imgs_path, captions_path, train=True):
        super(ImageCaptioningDataset).__init__()
        
        self.train = train
        self.img_paths = sorted(glob.glob(f"{imgs_path}/*.png"))
        
        df = pd.read_csv(captions_path, sep="\t")
        self.captions = []
        for i in range(len(df)):
            titles = []
            for j in range(1, 6):
                titles.append(to_ids((df.iloc[i, j])))
            self.captions.append(titles)

    def __getitem__(self, index):
        img = np.array(Image.open(self.img_paths[index]).convert("RGB"))
        if self.train:
            img = image_prepare(img)
        else:
            img = image_prepare_val(img)
        captions = random.choice(self.captions[index])
        return img, captions
    
    def __len__(self):
        return len(self.img_paths)

def collate_fn(batch):
    img_batch = torch.stack([el[0] for el in batch])
    local_max_seq_len = max(len(el[1]) for el in batch)
    captions_batch = torch.tensor([el[1] + [3] * (local_max_seq_len - len(el[1])) for el in batch])
    return img_batch, captions_batch

def get_val_dataloader(dataset, batch_size):
    dataloader_val = DataLoader(
    	dataset=dataset,
    	batch_size=batch_size,
    	collate_fn=collate_fn,
    	shuffle=False,
    	drop_last=False,
    )
    return dataloader_val
