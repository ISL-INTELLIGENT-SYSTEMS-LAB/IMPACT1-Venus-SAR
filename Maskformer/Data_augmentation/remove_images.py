import numpy as np
import os
from PIL import Image

def remove_images(image_dir, label_dir):
    """Remove images and masks that do not contain multiple values within the image."""
    # Get the list of masks
    masks = os.listdir(label_dir)
    
    listOfRemovedImages = []

    # Iterate through all mask images and remove the pairs that dont have multiple values within the image
    for maskp in masks:
        mask_path = os.path.join(label_dir, maskp)
        mask = Image.open(mask_path)
        mask = np.array(mask)
        
        numberOfUniqueValues = len(np.unique(mask))
        if numberOfUniqueValues == 1:
            # Remove the mask and the corresponding image
            image_path = os.path.join(image_dir, maskp)
            os.remove(mask_path)
            os.remove(image_path)
            listOfRemovedImages.append(maskp)
    
        
            
    print(f"Removed {len(listOfRemovedImages)} images and masks that do not contain any pixel values equal to (0,0,0).")
    # Print the list of removed images on separate lines
    print("The following pairs were removed:")
    for removedImage in listOfRemovedImages:
        print(removedImage)
    
    
    
    
# Remove images and labels that are not part of the dataset
remove_images("data\preprocessed_images", "data\preprocessed_labels")