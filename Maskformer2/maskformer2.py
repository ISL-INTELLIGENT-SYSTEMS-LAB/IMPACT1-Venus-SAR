# Modules for Mask2Former
from transformers import AutoProcessor, AutoModelForUniversalSegmentation

import numpy as np
from PIL import Image
import seaborn as sns
import warnings
import logging

import torch
from torch.utils.data import Dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve
print("Modules Imported")

# Suppress logging information from transformers
logging.getLogger("transformers").setLevel(logging.ERROR)

# Suppress potential warnings from imported packages
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class CustomDataset(Dataset):
    def __init__(self, processor):
        self.processor = processor

    def __getitem__(self, idx):
        # Load image
        image = Image.open(r"C:\Users\forth\Desktop\MaskFormer2\data\verified_images\0_-1_BL.png").convert("RGB")
        
        # Resize the image to a smaller size, e.g., 352x352
        image = image.resize((352, 352), Image.LANCZOS)

        # Load the grayscale semantic segmentation map
        mask = Image.open(r"C:\Users\forth\Desktop\MaskFormer2\data\verified_labels\0_-1_BL.png").convert("L")
        mask = mask.resize((352, 352), Image.NEAREST)
        mask = np.array(mask)

        # Debugging: Check the dimensions and unique labels
        #print(f"Image shape: {np.array(image).shape}")
        #print(f"Mask shape: {mask.shape}")
        #print(f"Unique labels in mask: {np.unique(mask)}")

        # Ensure the image and segmentation map have the correct dimensions
        assert image.size == (352, 352), "Image size mismatch!"
        assert mask.shape == (352, 352), "Segmentation map size mismatch!"

        # Use the processor to convert the image and segmentation map to tensors
        inputs = self.processor(images=image, segmentation_maps=[mask], task_inputs=["semantic"], return_tensors="pt")
        inputs = {k: v.squeeze() if isinstance(v, torch.Tensor) else v[0] for k, v in inputs.items()}

        return inputs

    def __len__(self):
        return 2

print("CustomDataset Class Defined")


def draw_semantic_segmentation(segmentation):
    # Move the segmentation tensor to CPU and convert to numpy
    segmentation = segmentation.cpu().numpy()

    # Ensure segmentation map has manageable dimensions
    if segmentation.ndim != 2:
        raise ValueError("Segmentation map should be 2D, got shape: {}".format(segmentation.shape))

    # Define the custom labels
    custom_labels = {
        0: "Background",
        1: "Wrinkle Ridge"
    }

    # Get the used color map
    viridis = cm.get_cmap('viridis', len(custom_labels))

    # Get all the unique labels in the segmentation map
    labels_ids = np.unique(segmentation).tolist()

    # Plot the segmentation map
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(segmentation, cmap=viridis)

    handles = []
    for label_id in custom_labels:
        label = custom_labels[label_id]
        color = viridis(label_id)
        handles.append(mpatches.Patch(color=color, label=label))

    ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(r"metrics\\segmentation_map.png")
    #plt.show()

def plot_confusion_matrix(y_true, y_pred, class_labels):
    # Define the custom labels
    custom_labels = ["Background", "Wrinkle Ridge"]

    # Compute confusion matrix (pixel counts)
    cm = confusion_matrix(y_true.flatten(), y_pred.flatten(), labels=class_labels)
    
    # Plot confusion matrix with pixel counts
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=custom_labels, yticklabels=custom_labels, ax=ax)
    ax.set_title('Confusion Matrix (Pixel Counts)')
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    plt.savefig(r"metrics\\confusion_matrix_pixel_counts.png")
    #plt.show()

    # Compute confusion matrix (percentages)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plot confusion matrix with percentages
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(cm_percentage, annot=True, fmt='.2%', cmap='Blues', xticklabels=custom_labels, yticklabels=custom_labels, ax=ax)
    ax.set_title('Confusion Matrix (Percentage)')
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    plt.savefig(r"metrics\\confusion_matrix.png")
    #plt.show()

def plot_confidence_matrix(confidences, y_true, y_pred, class_labels):
    # Define the custom labels
    custom_labels = ["Background", "Wrinkle Ridge"]

    confidence_matrix = np.zeros((len(class_labels), len(class_labels)))

    for i, (true_class, pred_class) in enumerate(zip(y_true.flatten(), y_pred.flatten())):
        confidence_matrix[true_class, pred_class] += confidences.flatten()[i]

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(confidence_matrix, annot=True, fmt='.2f', cmap='Blues', xticklabels=custom_labels, yticklabels=custom_labels, ax=ax)
    ax.set_title('Confidence Matrix')
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    plt.savefig(r"metrics\\confidence_matrix.png")
    #plt.show()

def plot_precision_recall_curve(y_true, y_pred_proba, class_labels):
    # Define the custom labels
    custom_labels = ["Background", "Wrinkle Ridge"]

    y_true_binarized = label_binarize(y_true.flatten(), classes=class_labels)
    
    for i, label in enumerate(custom_labels):
        precision, recall, _ = precision_recall_curve(y_true_binarized[:, i], y_pred_proba[:, i].flatten())
        plt.plot(recall, precision, lw=2, label=f'Class {label}')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='best')
    plt.savefig(r"metrics\\precision_recall_curve.png")
    #plt.show()
    
def plot_roc_curve(y_true, y_pred_proba, class_labels):
    # Define the custom labels
    custom_labels = ["Background", "Wrinkle Ridge"]

    y_true_binarized = label_binarize(y_true.flatten(), classes=class_labels)
    
    for i, label in enumerate(custom_labels):
        fpr, tpr, _ = roc_curve(y_true_binarized[:, i], y_pred_proba[:, i].flatten())
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'Class {label} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='best')
    plt.savefig(r"metrics\\roc_curve.png")
    #plt.show()

print("Functions Defined")


# Create the processor and model
processor = AutoProcessor.from_pretrained("shi-labs/oneformer_coco_swin_large")
processor.image_processor.size = (352, 352)
processor.image_processor.do_resize = False
model = AutoModelForUniversalSegmentation.from_pretrained("shi-labs/oneformer_coco_swin_large", is_training=True)
processor.image_processor.num_text = model.config.num_queries - model.config.text_encoder_n_ctx
print("Processor and Model Created")

# Create the dataset
dataset = CustomDataset(processor)
print("Dataset Created")

"""# Load the first example and print its shape
example = dataset[0]
for k, v in example.items():
    if isinstance(v, torch.Tensor):
        print(f"{k}: {v.shape}")"""

# Create the dataloader and load a batch of examples
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
batch = next(iter(dataloader))
"""for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        print(f"{k}: {v.shape}")"""
print("Dataloader Created")

# Verify the data
ADE_MEAN = np.array([123.675, 116.280, 103.530]) / 255
ADE_STD = np.array([58.395, 57.120, 57.375]) / 255

# Unnormalize the image
unnormalized_image = (batch["pixel_values"][0].squeeze().numpy() * np.array(ADE_STD)[:, None, None]) + np.array(ADE_MEAN)[:, None, None]
unnormalized_image = (unnormalized_image * 255).astype(np.uint8)
unnormalized_image = np.moveaxis(unnormalized_image, 0, -1)
# Display the image
colorimg = Image.fromarray(unnormalized_image)
#colorimg.show()
colorimg.save(r"metrics\\image.png")

# Display the mask
idx = 0
visual_mask = (batch["mask_labels"][0][idx].bool().numpy() * 255).astype(np.uint8)
maskimg = Image.fromarray(visual_mask)
#maskimg.show()
maskimg.save(r"metrics\\mask.png")

# Declare the optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Move the model to the device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train the model
model.train()
model.to(device)
for epoch in range(20):  # Loop over the dataset multiple times
    for batch in dataloader:
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Send batch to device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward pass
        outputs = model(**batch)

        # Backward pass + optimize
        loss = outputs.loss
        print("Loss:", loss.item())
        loss.backward()
        optimizer.step()

print("Finished Training")

# Switch to evaluation mode
model.eval()

# Set the is_training attribute of the base OneFormerModel to None after training
model.model.is_training = False

# Load the image
image = Image.open(r"C:\Users\forth\Desktop\MaskFormer2\data\verified_images\0_-1_BR.png")
image = image.resize((352, 352), Image.LANCZOS)

# Prepare image for the model
inputs = processor(images=image, task_inputs=["semantic"], return_tensors="pt")

# Check if CUDA is available and move model and inputs to the appropriate device
inputs = {k: v.to(device) for k, v in inputs.items()}

"""# Verify the shapes of the inputs
for k, v in inputs.items():
    if isinstance(v, torch.Tensor):
        print(f"{k}: {v.shape}")"""

# Forward pass (no need for gradients at inference time)
with torch.no_grad():
    outputs = model(**inputs)

# Apply softmax to the mask logits to get predicted probabilities
y_pred_proba = torch.softmax(outputs.masks_queries_logits, dim=1)

# Predicted segmentation map (argmax to get the most likely class per pixel)
semantic_segmentation = y_pred_proba.argmax(dim=1)  # Choose the class with the highest probability

# Resize the predicted segmentation to match the ground truth size
ground_truth_segmentation = Image.open(r"C:\Users\forth\Desktop\MaskFormer2\data\verified_labels\0_-1_BR.png")
ground_truth_segmentation = ground_truth_segmentation.resize((352, 352), Image.NEAREST)  # Ensure it matches the size of the input
ground_truth_segmentation = torch.tensor(np.array(ground_truth_segmentation), dtype=torch.long)

# Resize the predicted segmentation to match the ground truth size
semantic_segmentation_resized = torch.nn.functional.interpolate(
    semantic_segmentation.unsqueeze(1).float(),  # Add channel dimension
    size=ground_truth_segmentation.shape[-2:],  # Target size from ground truth
    mode="nearest"
).squeeze(1).long()  # Remove channel dimension and convert back to long tensor

# Ensure the segmentation map is 2D for visualization
semantic_segmentation_resized = semantic_segmentation_resized.squeeze(0)  # Remove the batch dimension

# Call the function to draw the segmentation map
draw_semantic_segmentation(semantic_segmentation_resized)

# Ensure both the true and predicted labels are available for comparison
true_labels = ground_truth_segmentation.cpu().numpy().flatten()
pred_labels = semantic_segmentation_resized.cpu().numpy().flatten()
confidences = y_pred_proba.max(dim=1)[0].cpu().numpy().flatten()

# List of class labels used in the model (e.g., [0, 1])
class_labels = [0, 1]

# Now, plot the metrics with custom labels
plot_confusion_matrix(true_labels, pred_labels, class_labels)
plot_confidence_matrix(confidences, true_labels, pred_labels, class_labels)
plot_precision_recall_curve(ground_truth_segmentation.cpu().numpy(), y_pred_proba.cpu().numpy(), class_labels)
plot_roc_curve(ground_truth_segmentation.cpu().numpy(), y_pred_proba.cpu().numpy(), class_labels)