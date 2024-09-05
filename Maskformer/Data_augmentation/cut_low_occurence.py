import os
import numpy as np
from PIL import Image

def cut_low_occurence(data, threshold=0.07):
    """
        Creates a list of image and mask pairs to remove from the dataset based on the percentage of black(0,0,0) pixels in the mask.
        
        Parameters:
            data (list): list of masks
            threshold (float | Default==0.07): the threshold to remove the image and mask pairs
            
        Returns:
            images_to_remove (list): list of images to remove
    """
    print('Calculating the percentage of black pixels in the masks...', end='')
    
    images_to_remove = []
    for mask in data:    
        maskimg = Image.open(mask)
        masknp = np.array(maskimg)
        num_pixels = 352 * 352
        num_black_pixels = np.sum(masknp == [0])
        percentage = num_black_pixels / num_pixels
        if percentage < threshold:
            images_to_remove.append(mask)
            
    print('Done!')
    return images_to_remove

def remove_pairs(pairs_to_remove):
    """
        Removes the image and mask pairs from the dataset.
        
        Parameters:
            pairs_to_remove (list): list of mask file names to remove (images are have the same name as the mask)
    """
    print('Removing the pairs...', end='')
    for img in pairs_to_remove:
        os.remove(img)
        img = img.replace('preprocessed_labels', 'preprocessed_images')
        os.remove(img)

def main():
    mask_path = '/home/lhernandez2/Venus_SAR/Dataset/preprocessed_labels'
    masks = os.listdir(mask_path)
    masks = [os.path.join(mask_path, mask) for mask in masks]
    
    images_to_remove = cut_low_occurence(masks)
    print(f'\nThere are {len(images_to_remove)} pairs to remove from the dataset\n')
    
    remove_pairs(images_to_remove)
    print('Done!')
    
if __name__ == '__main__':
    main()