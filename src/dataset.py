import os
import csv
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
from vocab import Vocabulary

class Flickr8kDataset(Dataset):
    def __init__(self, img_dir, captions_file, train_file=None, vocab=None, freq_threshold=5):
        self.img_dir = img_dir

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.imgs = []
        self.captions = []

        with open(captions_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                img_name = row["image"].strip()
                caption = row["caption"].strip()
                self.imgs.append(img_name)
                self.captions.append(caption)

        if vocab is None:
            self.vocab = Vocabulary(freq_threshold)
            self.vocab.build_vocab(self.captions)
        else:
            self.vocab = vocab

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        caption = self.captions[idx]
        tokens = [self.vocab.stoi["<start>"]]
        tokens += self.vocab.numericalize(caption)
        tokens += [self.vocab.stoi["<end>"]]
        caption_tensor = torch.tensor(tokens, dtype=torch.long)

        return image, caption_tensor


def collate_fn(batch):
    imgs, captions = zip(*batch)
    imgs = torch.stack(imgs, 0)

    lengths = [len(cap) for cap in captions]
    max_len = max(lengths)
    padded = torch.zeros(len(captions), max_len, dtype=torch.long)
    for i, cap in enumerate(captions):
        padded[i, :len(cap)] = cap

    return imgs, padded, torch.tensor(lengths)