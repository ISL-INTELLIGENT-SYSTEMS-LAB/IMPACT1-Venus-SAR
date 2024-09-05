from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt

# Paths to the example image and mask
image_path = r"data\images\0_-1.png"
mask_path = r"data\HLOD_masks\0_-1.png"

# Load the image and mask
image = Image.open(image_path).convert("RGB")
mask = Image.open(mask_path).convert("L")

# Convert to numpy arrays
image_array = np.array(image)
mask_array = np.array(mask)

print(f"Image shape: {image_array.shape}")
print(f"Mask shape: {mask_array.shape}")

# Ensure the mask has the same dimensions as the image
if mask_array.shape != image_array.shape[:2]:
    print(f"Resizing mask from {mask_array.shape} to {image_array.shape[:2]}")
    mask = mask.resize(image_array.shape[:2], Image.NEAREST)
    mask_array = np.array(mask)

print(f"Resized mask shape: {mask_array.shape}")

# Visualize the image and mask
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(mask, cmap="gray")
plt.title("Mask")
plt.axis("off")

plt.show()