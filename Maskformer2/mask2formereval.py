import os
import datetime
import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, AutoModelForUniversalSegmentation
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, jaccard_score, roc_curve, auc
import torch.nn.functional as F
import collections
from sklearn.metrics import precision_score, recall_score

SEED = random.randint(1000, 9999)
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
print(f"Random Seed: {SEED}")

# Directories
test_img_dir = "test_images"
test_mask_dir = "test_mask"
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
mdl_results_dir = os.path.join("results_dir", f'maskformer_test_eval_{timestamp}_{SEED}')
MODEL_DIR = "model_dir"
MODEL_PATH = os.path.join(MODEL_DIR, "model.safetensors")
os.makedirs(mdl_results_dir, exist_ok=True)

RESIZE_DIM = (512, 512)  # Resize to whatever dimension
BATCH_SIZE = 1  # Adjust batch size accordingly

# Create Dataset
class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, processor, resize=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.processor = processor
        self.resize = resize
        self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
        self.mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith('.png')])
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

        return inputs, mask, np.array(image)

# Save visualization
def save_visualization(image, predicted_mask, ground_truth_mask, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(predicted_mask, cmap='gray')
    axes[0].set_title("Segmentation Mask")
    axes[0].axis("off")
    
    axes[1].imshow(ground_truth_mask.squeeze(0), cmap='gray')
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")
    
    axes[2].imshow(image)
    axes[2].set_title("Original Image")
    axes[2].axis("off")
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization: {save_path}")

processor = AutoProcessor.from_pretrained("Shi-Labs/oneformer_coco_swin_large")

# Load model
model = AutoModelForUniversalSegmentation.from_pretrained(MODEL_DIR)
model.to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

# Ensure num_text is set
if hasattr(processor.image_processor, "num_text") and processor.image_processor.num_text is None:
    processor.image_processor.num_text = model.config.num_queries - model.config.text_encoder_n_ctx

test_dataset = CustomDataset(test_img_dir, test_mask_dir, processor, resize=RESIZE_DIM)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

def evaluate_model(model, dataloader, device, output_dir):
    model.eval()
    all_preds, all_gts = [], []
    threshold = 0.5
    
    with torch.no_grad():
        for batch_idx, (batch, masks, images) in enumerate(tqdm(dataloader, desc="Evaluating Model")):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch, output_hidden_states=True)
            logits = outputs["masks_queries_logits"][..., 0, :, :]
            preds_resized = F.interpolate(logits.unsqueeze(1), size=masks.shape[1:], mode="bilinear", align_corners=False).squeeze(1)
            preds = (torch.sigmoid(preds_resized) > threshold).int().cpu().numpy()
            all_preds.extend(preds.flatten())
            all_gts.extend(masks.flatten())
            
            save_visualization(images[0], preds[0], masks, output_dir, f"result_{batch_idx}.png")
    
    all_preds = np.array(all_preds, dtype=np.uint8)
    all_gts = np.array(all_gts, dtype=np.uint8)
    acc = round(accuracy_score(all_gts, all_preds), 4)
    precision = np.round(precision_score(all_gts, all_preds, average='macro', zero_division=0), 4)
    recall = np.round(recall_score(all_gts, all_preds, average='macro', zero_division=0), 2)
    mean_iou = round(jaccard_score(all_gts, all_preds, average='macro'), 4)
    
    print(f"Evaluation Complete:\n"
          f" Accuracy: {acc}\n"
          f" IoU: {mean_iou}\n"
          f" Precision: {precision}\n"
          f" Recall: {recall}")
    return acc, precision, recall, mean_iou

evaluation_results = evaluate_model(model, test_loader, model.device, mdl_results_dir)
print('Evaluation complete')
