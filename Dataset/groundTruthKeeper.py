import cv2
import numpy as np
import os
from tqdm import tqdm

def mask_cutter(img_path, keep_path, save_path, dialation=5):
    # Load the thresholded image and the Keep image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    keep = cv2.imread(keep_path, cv2.IMREAD_COLOR)

    img = cv2.resize(img, (keep.shape[1], keep.shape[0]))

    # Create a binary mask for black pixels in the Keep image
    black_mask = np.all(keep == [0, 0, 0], axis=-1).astype(np.uint8) * 255  # Mask for black pixels

    # Expand the black pixel mask by a 50-pixel radius using dilation
    kernel = np.ones((dialation*2+1, dialation*2+1), np.uint8)
    dilated_mask = cv2.dilate(black_mask, kernel, iterations=1)

    # Apply the mask to keep the areas inside the black mask and save the image
    img[dilated_mask == 0] = 0
    cv2.imwrite(save_path, img)

def clean_cuts(img_path, save_path):
    # Load the image & convert to grayscale
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Remove noise from the image using Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Otsu's thresholding to the blurred image and find contours
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a mask for the contours and draw them on the mask
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, contours, -1, (255, 255, 255), -1)

    # Apply the mask to the original image and save it
    img[mask == 0] = 0
    cv2.imwrite(save_path, img)


if __name__ == '__main__':
    # Define the paths to the images
    img_path = 'verified_pairs/HLOD_masks/'
    keep_path = 'verified_pairs/masks/'

    # Cut the mask using the Keep image
    for img in tqdm(os.listdir(img_path), desc="Processing masks"):
        mask_cutter(f'{img_path}{img}', f'{keep_path}{img}', f'{img_path}output_{img}')
        clean_cuts(f'{img_path}output_{img}', f'{img_path}{img}')
        os.remove(f'{img_path}output_{img}')

    print('All masks processed successfully')
