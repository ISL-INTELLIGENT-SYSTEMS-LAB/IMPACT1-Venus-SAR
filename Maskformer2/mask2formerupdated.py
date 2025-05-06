import os
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import random
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoModelForUniversalSegmentation
from tqdm import tqdm
from PIL import Image
import numpy as np
import warnings
import logging
import matplotlib.pyplot as plt

print("Modules Imported")

# Suppress logging information from transformers
logging.getLogger("transformers").setLevel(logging.ERROR)

# Suppress potential warnings from imported packages
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Set random seed for reproducibility
SEED = random.randint(1000, 9999)
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
print(f"Random Seed: {SEED}")

# Resize option
RESIZE_DIM = (512, 512)  # Adjust resize dimensions as needed

#### DATASET ####
class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, processor, resize=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.processor = processor
        self.resize = resize
        self.image_files = sorted ([f for f in os.listdir(image_dir) if f.endswith('.png')])
        self.mask_files = sorted ([f for f in os.listdir(mask_dir) if f.endswith('.png')])
        assert len(self.image_files) == len(self.mask_files), "Mismatch between number of images and masks."

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.resize:
            image = image.resize(self.resize, Image.LANCZOS)
            mask = mask.resize(self.resize, Image.NEAREST)

        mask = np.array(mask)

        inputs = self.processor(images=image, segmentation_maps=[mask], task_inputs=["semantic"], return_tensors="pt")
        inputs = {k: v.squeeze() if isinstance(v, torch.Tensor) else v[0] for k, v in inputs.items()}

        return inputs

print("Dataset Class Defined")

#### SAVE FUNCTIONS ####
def save_model(model, save_directory):
    os.makedirs(save_directory, exist_ok=True)
    model.save_pretrained(save_directory)
    print(f"Model and config saved to {save_directory}")

def save_loss_plot(train_losses, val_losses, save_directory):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    os.makedirs(save_directory, exist_fok=True)
    plt.savefig(os.path.join(save_directory, "loss_plot.png"))
    plt.close()
    print(f"Loss plot saved to {os.path.join(save_directory, 'loss_plot.png')}")

#### INITIALIZATION ####
model = AutoModelForUniversalSegmentation.from_pretrained(
    "model_path",
    is_training=True
)

# Load the processor
processor = AutoProcessor.from_pretrained(
    "model_path"
)

# Ensure num_text is correctly set using the model configuration (Debugging)
if hasattr(model.config, "num_queries") and hasattr(model.config, "text_encoder_n_ctx"):
    processor.image_processor.num_text = model.config.num_queries - model.config.text_encoder_n_ctx
else:
    raise AttributeError("The model config does not have 'num_queries' or 'text_encoder_n_ctx'")
print("Processor and Model Created")

#### DATA LOADING ####
train_image_dir = "train_images"
train_mask_dir = "train_label"
val_image_dir = "val_image"
val_mask_dir = "val_label"

BATCH_SIZE = 2  # Adjust batch size as needed

train_dataset = CustomDataset(train_image_dir, train_mask_dir, processor, resize=RESIZE_DIM)
val_dataset = CustomDataset(val_image_dir, val_mask_dir, processor, resize=RESIZE_DIM)
print("Training and Validation Datasets Created")

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
print("Dataloaders Created")

#### TRAINING SETUP ####
optimizer = optim.Adam(model.parameters(), lr=1e-5)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

current_date = datetime.datetime.now().strftime("%Y-%m-%d")
save_dir = os.path.join('result_dir',f"maskform2_{current_date}_{SEED}")
os.makedirs(save_dir, exist_ok=True)

num_epochs = 30
best_val_loss = float('inf')
train_losses = []
val_losses = []

#### TRAINING LOOP ####
model.train()
model.to(device)
for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    
    train_loss = 0.0
    for batch in tqdm(train_dataloader, desc='Training', leave=False):
        optimizer.zero_grad()
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_dataloader)
    train_losses.append(avg_train_loss)
    
    model.eval()
    val_loss = 0.0
    for batch in tqdm(val_dataloader, desc='Validation', leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        val_loss += outputs.loss.item()
    
    avg_val_loss = val_loss / len(val_dataloader)
    val_losses.append(avg_val_loss)
    
    print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f} | Validation Loss: {avg_val_loss:.4f}')
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        save_model(model, save_dir)

save_loss_plot(train_losses, val_losses, save_dir)
print("Finished Training")
