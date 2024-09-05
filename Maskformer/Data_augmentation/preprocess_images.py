import os
from PIL import Image
import numpy as np
import concurrent.futures
from tqdm import tqdm  # Importing tqdm to create a progress bar
import yaml

def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def preprocess_image(image_path, label_path, output_images_dir, output_labels_dir):
    """Preprocess an image and its corresponding label."""
    # Open the label file and convert to NumPy array
    mask = Image.open(label_path)
    mask = np.array(mask)

    # Duplicate the single channel value across all three channels
    mask = np.dstack((mask, mask, mask))

    # Save the preprocessed mask
    mask_filename = os.path.basename(label_path)
    mask = Image.fromarray(mask.astype(np.uint8), 'RGB')

    # Split the image into 4 quadrants and resize them to 352x352
    width, height = mask.size
    quadrants = {
        "TL": mask.crop((0, 0, width // 2, height // 2)).resize((352, 352)),
        "TR": mask.crop((width // 2, 0, width, height // 2)).resize((352, 352)),
        "BL": mask.crop((0, height // 2, width // 2, height)).resize((352, 352)),
        "BR": mask.crop((width // 2, height // 2, width, height)).resize((352, 352))
    }

    # Save the quadrants and append the quadrant name to the filename
    for key, quadrant in quadrants.items():
        quadrant.save(os.path.join(output_labels_dir, mask_filename.replace(".png", f"_{key}.png")))

    # Process the corresponding image in a similar manner
    image_filename = os.path.basename(image_path)
    image = Image.open(image_path)
    image = np.array(image)
    image = np.dstack((image, image, image))
    image = Image.fromarray(image.astype(np.uint8), 'RGB')
    width, height = image.size
    quadrants = {
        "TL": image.crop((0, 0, width // 2, height // 2)).resize((352, 352)),
        "TR": image.crop((width // 2, 0, width, height // 2)).resize((352, 352)),
        "BL": image.crop((0, height // 2, width // 2, height)).resize((352, 352)),
        "BR": image.crop((width // 2, height // 2, width, height)).resize((352, 352))
    }

    # Save the quadrants and append the quadrant name to the filename
    for key, quadrant in quadrants.items():
        quadrant.save(os.path.join(output_images_dir, image_filename.replace(".png", f"_{key}.png")))

def preprocess_images(images_dir, labels_dir, output_images_dir, output_labels_dir):
    """Preprocess all images in the specified directories."""
    # Create output directories if they don't exist
    if not os.path.exists(output_images_dir):
        os.makedirs(output_images_dir)
    if not os.path.exists(output_labels_dir):
        os.makedirs(output_labels_dir)

    # List all image files in the directory
    image_filenames = [f for f in os.listdir(images_dir) if f.endswith(".png")]

    # Use ThreadPoolExecutor to process images in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        # Create a progress bar
        with tqdm(total=len(image_filenames), desc="Processing images") as pbar:
            for filename in image_filenames:
                image_path = os.path.join(images_dir, filename)
                label_path = os.path.join(labels_dir, filename.replace(".png", ".png"))
                futures.append(executor.submit(preprocess_image, image_path, label_path, output_images_dir, output_labels_dir))

            # Update progress bar as tasks complete
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    print(f'Generated an exception: {exc}')
                finally:
                    pbar.update(1)

if __name__ == "__main__":
    # Load configuration
    config = load_config('/home/lhernandez2/Venus_SAR/Maskformer/Data_augmentation/preprocess_images_config.yaml')

    images_dir = config['images_dir']
    labels_dir = config['labels_dir']
    output_images_dir = config['output_images_dir']
    output_labels_dir = config['output_labels_dir']

    preprocess_images(images_dir, labels_dir, output_images_dir, output_labels_dir)
