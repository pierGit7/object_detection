import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import json
from torchvision import transforms as T

class PotholeProposalsDataset(Dataset):
    def __init__(self, img_dir, proposals_json, transform=None, split="train"):
        """
        img_dir: folder containing all images
        proposals_json: path to JSON file with labeled proposals
        transform: torchvision transforms applied to the image
        split: 'train', 'test', 'valid'
        """
        self.img_dir = img_dir
        self.transform = transform

        # Load JSON proposals
        with open(proposals_json, "r") as f:
            self.data = json.load(f)

        # ---- SPLIT DATA ----
        image_ids = sorted(list(self.data.keys()))
        n = len(image_ids)
        train_end = int(0.7 * n)
        test_end  = int(0.85 * n)

        if split == "train":
            self.image_ids = image_ids[:train_end]
        elif split == "test":
            self.image_ids = image_ids[train_end:test_end]
        elif split in ["valid", "val"]:
            self.image_ids = image_ids[test_end:]
        else:
            raise ValueError("split must be 'train', 'test', or 'valid'")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = os.path.join(self.img_dir, image_id)
        img = Image.open(img_path).convert("RGB")  # PIL Image

        # Apply transforms to the **full image**
        if self.transform:
            img = self.transform(img)
        else:
            img = T.ToTensor()(img)

        # Get proposals for this image
        proposals = self.data[image_id]
        boxes = []
        labels = []

        for prop in proposals:
            x1, y1, x2, y2, label = prop
            # Ensure valid box
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                # Faster R-CNN expects labels starting from 1 for foreground
                labels.append(int(label) + 1 if label > 0 else 0)  

        # Convert to tensors
        if len(boxes) == 0:
            # No proposals, add dummy box to avoid crashing
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels
        }

        return img, target
