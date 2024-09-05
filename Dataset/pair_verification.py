import os
import shutil


def pair_verification():
    """
    This function verifies the pairs of images and masks and copies them to a new directory.
    """
    images, masks = [], []

    # Extracting the longitude and latitude from the image file names
    for image in os.listdir('/home/lhernandez2/Venus_SAR/Dataset/images'):
        img = image.split('_pro')[0].split('_')
        img.append(f'{image}')
        images.append(img)

    # Extracting the longitude and latitude from the mask file names
    for mask in os.listdir('/home/lhernandez2/Venus_SAR/Dataset/masks'):
        msk = mask.split('.png')[0].split('lonlat_')[1].split('_')
        msk.append(f'{mask}')
        masks.append(msk)

    # Copying the verified pairs to a new directory
    for img in images: 
        for msk in masks:
            if img[0] == msk[0] and img[1] == msk[1]:
                shutil.copy(f'/home/lhernandez2/Venus_SAR/Dataset/images/{img[2]}', f'/home/lhernandez2/Venus_SAR/Dataset/verified_pairs/images/{img[0]}_{img[1]}.png')
                shutil.copy(f'/home/lhernandez2/Venus_SAR/Dataset/masks/{msk[2]}', f'/home/lhernandez2/Venus_SAR/Dataset/verified_pairs/masks/{msk[0]}_{msk[1]}.png')
                print(f'Copied image and masks for longitude {img[0]} and latitude {img[1]}')
            

if __name__ == '__main__':
    pair_verification()