import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
import glob
from sklearn.metrics import precision_score, recall_score, jaccard_score, roc_auc_score, confusion_matrix, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from transformers import SegformerForSemanticSegmentation
import datetime
import random
import datetime, random

# Configurations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "model_path"
MODEL_NAME = "model_name"
IMG_SIZE = (512, 512) # Adjust according to what JPL Folks want
BATCH_SIZE = 8
THRESHOLD = 0.25 # Adjust according to results: Higher, less detections more precise, Lower, more detections less precise
RESULTS_DIR = "results_dir"

# Seed and Additional Directory
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
random_seed = random.randint(1000, 9999)
SAVE_DIR = os.path.join(RESULTS_DIR, f"eval_{timestamp}_{random_seed}")
os.makedirs(SAVE_DIR, exist_ok=True)

TEST_IMG_DIR = "/home/lhernandez2/Venus_SAR/Dataset/verified_pairs/test_images_V322019/"
TEST_MASK_DIR = "/home/lhernandez2/Venus_SAR/Dataset/verified_pairs/test_labels_V322019/"

# Dataset Class
class CustomDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")

        if self.transform:
            img = self.transform(img)
            mask = self.transform(mask)

        mask = (mask > 0).long()
        return img, mask

# Transformations
transform = transforms.Compose([
    transforms.ToTensor(),
])

test_dataset = CustomDataset(TEST_IMG_DIR, TEST_MASK_DIR, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = SegformerForSemanticSegmentation.from_pretrained(
    MODEL_NAME, num_labels=2, ignore_mismatched_sizes=True, local_files_only=True
).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# Containers for evaluation metrics
all_preds = []
all_labels = []
all_probs = []
all_images = []

with torch.no_grad():
    for images, masks in test_loader:
        images, masks = images.to(device), masks.to(device)
        outputs = model(pixel_values=images).logits

        probs = torch.softmax(outputs, dim=1)[:, 1, :, :]
        preds = (probs > THRESHOLD).long()

        # Upsample predictions to original resolution (512x512)
        preds_resized = F.interpolate(
            preds.unsqueeze(1).float(), size=IMG_SIZE, mode='nearest'
        ).squeeze(1).long()

        # Upsample probabilities for metrics
        probs_resized = F.interpolate(
            probs.unsqueeze(1), size=IMG_SIZE, mode='bilinear', align_corners=False
        ).squeeze(1)

         # Ensure masks have channel dimension
        if masks.ndim == 3:
            masks = masks.unsqueeze(1)

        # Resize masks to match ground truth (512x512)
        masks_resized = F.interpolate(masks.float(), size=IMG_SIZE, mode="nearest").squeeze(1).long()

        # Resize predictions to match 512x512 (Debugging)
        #preds_resized = F.interpolate(probs.unsqueeze(1), size=IMG_SIZE, mode="bilinear", align_corners=False)
        #preds_resized = (preds_resized.squeeze(1) > THRESHOLD).long()

        #print(f"preds_resized shape: {preds_resized.shape}") (Debugging)
        #print(f"masks_resized shape: {masks_resized.shape}")
        #print(f"probs_resized shape: {probs_resized.shape}")
        
        all_preds.append(preds_resized.cpu().numpy())
        all_labels.append(masks_resized.cpu().numpy())
        all_probs.append(probs_resized.cpu().numpy())
        all_images.append(images.cpu().numpy()) 

# Flatten for metrics
y_true = np.concatenate([label.flatten() for label in all_labels])
y_pred = np.concatenate([pred.flatten() for pred in all_preds])
y_probs = np.concatenate([prob.flatten() for prob in all_probs])

# Compute metrics
mean_iou = jaccard_score(y_true, y_pred, average="binary", zero_division=0)
precision = precision_score(y_true, y_pred, average="binary", zero_division=0)
recall = recall_score(y_true, y_pred, average="binary", zero_division=0)
auc_score = roc_auc_score(y_true, y_probs)
overall_accuracy = accuracy_score(y_true, y_pred)

cm = confusion_matrix(y_true, y_pred)
per_class_accuracy = cm.diagonal() / cm.sum(axis=1)

# Display metrics
print("Evaluation Metrics:")
print(f"Mean IoU           : {mean_iou:.4f}")
print(f"Precision          : {precision:.4f}")
print(f"Recall             : {recall:.4f}")
print(f"AUC Score          : {auc_score:.4f}")
print(f"Overall Accuracy   : {overall_accuracy:.4f}")
print(f"Per-class Accuracy : {per_class_accuracy}")

# Save visualization (corrected dimension handling)
def save_visualization(pred, true, img, idx, save_dir):
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))

    ax[0].imshow(pred, cmap='gray')
    ax[0].set_title("Predicted Mask")
    ax[0].axis('off')

    ax[1].imshow(true.squeeze(), cmap='gray')
    ax[1].set_title("Ground Truth")
    ax[1].axis('off')

    ax[2].imshow(img.permute(1, 2, 0).cpu())
    ax[2].set_title("Original Image")
    ax[2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"result_{idx}.png"), bbox_inches='tight')
    plt.close()

# Create results folder with date and random seed
date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
random_seed = random.randint(1000, 9999)
visualization_dir = os.path.join(RESULTS_DIR, f"visualizations_{date_str}_{random_seed}")
os.makedirs(visualization_dir, exist_ok=True)

# Generate and save visualizations
idx = 0
for batch_preds, batch_trues, batch_imgs in zip(all_preds, all_labels, all_images):
    for i in range(batch_preds.shape[0]):
        pred = batch_preds[i]
        true = batch_trues[i]
        img = batch_imgs[i]
        save_visualization(pred, true, torch.tensor(img), idx, visualization_dir)
        idx += 1

print(f"Visualizations saved in: {visualization_dir}")
