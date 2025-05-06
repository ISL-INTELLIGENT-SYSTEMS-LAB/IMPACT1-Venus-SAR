import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation
import torch.nn.functional as F
from PIL import Image
import glob
import datetime
import random

# Configurations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "model_directory"
BATCH_SIZE = 50
EPOCHS = 200
IMG_SIZE = (512, 512)
SAVE_BASE_DIR = "save_directory"

# Seed and Output Dir
date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
random_seed = random.randint(1000, 9999)
SAVE_DIR = os.path.join(SAVE_BASE_DIR, f"segformer_train_{date_str}_{random_seed}")
os.makedirs(SAVE_DIR, exist_ok=True)

# Dataset Class
class CustomDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB").resize(IMG_SIZE, Image.BILINEAR)
        mask = Image.open(self.mask_paths[idx]).convert("L").resize(IMG_SIZE, Image.NEAREST)

        img = transforms.ToTensor()(img)
        mask = np.array(mask)
        mask = np.where(mask > 0, 1, 0).astype(np.uint8)
        mask = torch.from_numpy(mask).long()

        return img, mask

# Load Datasets
train_dataset = CustomDataset(
    "train_image",
    "train_label"
)
val_dataset = CustomDataset(
    "val_image",
    "val_label"
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load SegFormer Model
model = SegformerForSemanticSegmentation.from_pretrained(
    MODEL_PATH, num_labels=2, ignore_mismatched_sizes=True
).to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-5)

# Loss Functions 
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)[:, 1, :, :]
        targets = (targets == 1).float()
        intersection = (probs * targets).sum(dim=(1, 2))
        union = probs.sum(dim=(1, 2)) + targets.sum(dim=(1, 2))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.7, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
        
#Pick and choose whatever loss function to call
dice_loss = DiceLoss()
ce_loss = nn.CrossEntropyLoss(weight=torch.tensor([0.2, 0.8]).to(device)) # Adjust weights as needed (Fault 0: Fault 1)
#focal_loss = FocalLoss(alpha=0.9, gamma=2) # Adjust alpha and gamma as needed

def combined_loss(logits, targets):
    # Comment out/in depending on what loss you are using
    return ce_loss(logits, targets) + dice_loss(logits, targets)
    #return focal_loss(logits, targets) + dice_loss(logits, targets)

# Training Function
def train():
    best_val_loss = float("inf")
    train_losses, val_losses = [], []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        tqdm_loader = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for images, masks in tqdm_loader:
            images, masks = images.to(device), masks.long().to(device)

            optimizer.zero_grad()
            outputs = model(pixel_values=images).logits

            if masks.shape[1:] != outputs.shape[2:]:
                masks = F.interpolate(masks.unsqueeze(1).float(), size=outputs.shape[2:], mode="nearest").squeeze(1).long()

            loss = combined_loss(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            tqdm_loader.set_postfix(train_loss=loss.item())

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation Step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.long().to(device)
                outputs = model(pixel_values=images).logits

                if masks.shape[1:] != outputs.shape[2:]:
                    masks = F.interpolate(masks.unsqueeze(1).float(), size=outputs.shape[2:], mode="nearest").squeeze(1).long()

                loss = combined_loss(outputs, masks)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pth"))
            print(f"Saved Best Model in: {SAVE_DIR}")

    plt.plot(range(1, EPOCHS+1), train_losses, label="Train Loss")
    plt.plot(range(1, EPOCHS+1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(SAVE_DIR, "loss_plot.png"))
    plt.close()

if __name__ == "__main__":
    train()
  
