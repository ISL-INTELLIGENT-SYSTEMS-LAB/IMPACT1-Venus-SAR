import os
import shutil
import yaml

def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def move_images(source_img_dir, dest_img_dir, source_label_dir, dest_label_dir, images, labels):
    """Move images and their corresponding labels to new directories."""
    for image in images:
        for label in labels:
            if image.split('.')[0] == label.split('.')[0]:
                # Move image and label if they have matching filenames (excluding extension)
                shutil.move(os.path.join(source_img_dir, image), os.path.join(dest_img_dir, image))
                shutil.move(os.path.join(source_label_dir, label), os.path.join(dest_label_dir, label))
                break

def main():
    # Load configuration
    config = load_config('move_images_config.yaml')
    
    source_img_dir = config['source_img_dir']
    dest_img_dir = config['dest_img_dir']
    source_label_dir = config['source_label_dir']
    dest_label_dir = config['dest_label_dir']
    
    # Create destination directories if they don't exist
    os.makedirs(dest_img_dir, exist_ok=True)
    os.makedirs(dest_label_dir, exist_ok=True)
    
    images = os.listdir(source_img_dir)
    labels = os.listdir(source_label_dir)
    
    # Move images and labels according to the configuration
    move_images(source_img_dir, dest_img_dir, source_label_dir, dest_label_dir, images, labels)

if __name__ == "__main__":
    main()
