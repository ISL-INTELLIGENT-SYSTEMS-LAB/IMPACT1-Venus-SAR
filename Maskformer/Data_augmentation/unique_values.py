import numpy as np
import os
from PIL import Image


def count_unique_labels(label_dir):
    """Count the number of unique pixel values in the whole dataset."""
    # Get the list of masks
    masks = os.listdir(label_dir)
    
    # Number of unique values
    uniqueValues = []
    
    # Iterate through all mask images and count the number of unique pixel values
    for maskp in masks:
        mask_path = os.path.join(label_dir, maskp)
        mask = Image.open(mask_path)
        mask = np.array(mask)
        
        numberOfUniqueValues = np.unique(mask)
        uniqueValues.append(numberOfUniqueValues)
        
    return uniqueValues
    

def clean_pixel_values(label_dir):
    """Clean the pixel values in the masks."""
    # Get the list of masks
    masks = os.listdir(label_dir)
    
    # Iterate through all mask images and clean the pixel values
    for maskp in masks:
        mask_path = os.path.join(label_dir, maskp)
        mask = Image.open(mask_path)
        mask = np.array(mask)
        
        # Clean the pixel values (any value greater than 50 will be set to 255 and the rest to 0)
        mask[mask <= 50] = 1
        mask[mask > 50] = 0
        
        print(np.unique(mask))
        
        # Save the cleaned mask
        mask = Image.fromarray(mask)
        mask.save(mask_path)
        
        
    
    
        
"""# Count the number of unique pixel values in each mask image
precorrection = count_unique_labels("data\preprocessed_labels")
print(f"Number of unique pixel values in the dataset: {np.unique(precorrection)}")"""

clean_pixel_values("/home/lhernandez2/Venus_SAR/Dataset/preprocessed_labels")
"""postcorrection = count_unique_labels("data\preprocessed_labels")
print(f"Number of unique pixel values in the dataset after correction: {np.unique(postcorrection)}")"""
