import os
from PIL import Image
from tqdm import tqdm

def resize_image_and_mask(image_path, mask_path, old_size=(1407, 1407), new_size=(1408, 1408)):
    # Resize image
    with Image.open(image_path) as img:
        if img.size == old_size:
            img_resized = img.resize(new_size, Image.LANCZOS)
            img_resized.save(image_path)
        else:
            print(f"Skipping {image_path}, size is not {old_size}.")

    # Resize mask
    with Image.open(mask_path) as mask:
        if mask.size == old_size:
            mask_resized = mask.resize(new_size, Image.LANCZOS)
            mask_resized.save(mask_path)
        else:
            print(f"Skipping {mask_path}, size is not {old_size}.")

def resize_images_and_masks(image_directory, mask_directory, old_size=(1407, 1407), new_size=(1408, 1408)):
    # Ensure the directories exist
    if not os.path.exists(image_directory):
        print(f"Image directory {image_directory} does not exist.")
        return
    if not os.path.exists(mask_directory):
        print(f"Mask directory {mask_directory} does not exist.")
        return

    # Get a list of all image files in the directory
    image_files = [f for f in os.listdir(image_directory) if os.path.isfile(os.path.join(image_directory, f)) and f.lower().endswith(('png', 'jpg', 'jpeg', 'bmp', 'tiff', 'gif'))]

    # Process each image file and its corresponding mask
    for image_file in tqdm(image_files, desc="Resizing images and masks"):
        image_path = os.path.join(image_directory, image_file)
        mask_path = os.path.join(mask_directory, image_file)  # Assuming masks have the same filenames as images

        try:
            resize_image_and_mask(image_path, mask_path, old_size, new_size)
        except Exception as e:
            print(f"Error processing {image_path} and {mask_path}: {e}")

# Directories containing images and masks
image_directory = "/home/lhernandez2/Venus_SAR/Dataset/verified_pairs/images"
mask_directory = "/home/lhernandez2/Venus_SAR/Dataset/verified_pairs/masks"

resize_images_and_masks(image_directory, mask_directory)