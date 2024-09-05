import os
import numpy as np
from PIL import Image

# Define the directories where the images and masks are stored
image_dir = r"C:\Users\forth\Desktop\MaskFormer2\data\preprocessed_images"
mask_dir = r"C:\Users\forth\Desktop\MaskFormer2\data\preprocessed_labels"

# Define the target size
target_size = (352, 352)

# List all files in the directories
image_files = sorted(os.listdir(image_dir))
mask_files = sorted(os.listdir(mask_dir))

# Ensure that the number of images matches the number of masks
assert len(image_files) == len(mask_files), "Number of images and masks do not match!"

# Iterate over each image and mask pair
for image_file, mask_file in zip(image_files, mask_files):
    # Load the image and mask
    image_path = os.path.join(image_dir, image_file)
    mask_path = os.path.join(mask_dir, mask_file)
    
    image = Image.open(image_path)
    mask = Image.open(mask_path)

    # Resize image and mask
    image = image.resize(target_size, Image.LANCZOS)
    
    # Convert the mask to grayscale if it has multiple channels
    if mask.mode != 'L':
        mask = mask.convert('L')
    
    mask = mask.resize(target_size, Image.NEAREST)

    # Save the resized image and mask (overwriting the originals or in a new directory)
    image.save(r"C:\Users\forth\Desktop\MaskFormer2\data\verified_images\\" + image_file)
    mask.save(r"C:\Users\forth\Desktop\MaskFormer2\data\verified_labels\\" + mask_file)

    # Debugging: Print the shapes to ensure they match
    print(f"Processed {image_file} and {mask_file}: {image.size} == {mask.size}")
    
    imagenp = np.array(image)
    masknp = np.array(mask)
    print(f"Processed {image_file} and {mask_file}: {imagenp.shape} == {masknp.shape}\n")

print("All images and masks have been processed and resized.")
