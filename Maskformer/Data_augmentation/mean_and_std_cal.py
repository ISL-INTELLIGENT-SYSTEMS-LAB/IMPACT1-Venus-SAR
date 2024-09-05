import os
import numpy as np
from PIL import Image
from tqdm.auto import tqdm

def calculate_mean_std(image_dir):
    # Initialize sum and sum of squares
    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_squared_sum = np.zeros(3, dtype=np.float64)
    num_pixels = 0

    # Iterate over all images in the directory
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff', 'gif'))]

    for img_file in tqdm(image_files, desc="Calculating mean and std"):
        img_path = os.path.join(image_dir, img_file)
        with Image.open(img_path) as img:
            img = np.array(img.convert('RGB')) / 255.0  # Normalize to [0, 1]
            pixel_sum += img.sum(axis=(0, 1))
            pixel_squared_sum += (img ** 2).sum(axis=(0, 1))
            num_pixels += img.shape[0] * img.shape[1]

    # Calculate mean and std
    mean = pixel_sum / num_pixels
    std = np.sqrt((pixel_squared_sum / num_pixels) - (mean ** 2))

    return mean, std

# Example usage
image_directory = 'C:\\Users\\forth\\Desktop\\MaskFormer\\data\\preprocessed_images'
mean, std = calculate_mean_std(image_directory)
print(f"ADE_MEAN: {mean.tolist()}")
print(f"ADE_STD: {std.tolist()}")