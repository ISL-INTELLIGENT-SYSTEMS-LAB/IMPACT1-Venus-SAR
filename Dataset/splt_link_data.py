"""
Venus_SAR Data Splitter.

This script is designed to split images and their corresponding masks into
training, validation, and testing datasets for further machine learning tasks.
It uses predefined directory structures for verified images and labels, and
organizes the data into appropriate subdirectories for training, validation,
and testing.

Directory structure:
- Input:
    - verified_images (raw images)
    - verified_labels (corresponding masks/labels for the images)
- Output:
    - training_images (images for training)
    - training_masks (masks for training)
    - validation_images (images for validation)
    - validation_masks (masks for validation)
    - testing_images (images for testing)
    - testing_masks (masks for testing)

The script assumes that the dataset is already present and organized, and the
`split_data`function is responsible for splitting the data into the appropriate
directories.
"""

import os
from sklearn.model_selection import train_test_split as tts


def create_dir(path):
    """
    Create a directory if it does not exist.

    Parameters:
    - path (str): The path of the directory to be created.

    This function checks if the specified directory exists. If it doesn't,
    it creates the directory.
    """
    if not os.path.exists(path):
        os.makedirs(path)


def split_data(
    image_dir, mask_dir,
    train_img_dir, train_mask_dir,
    val_img_dir, val_mask_dir,
    test_img_dir, test_mask_dir
):
    """
    Split images and masks into training, validation, and test sets.

    Parameters:
    - image_dir (str): Directory containing the images.
    - mask_dir (str): Directory containing the corresponding masks.
    - train_img_dir (str): Directory to store training images.
    - train_mask_dir (str): Directory to store training masks.
    - val_img_dir (str): Directory to store validation images.
    - val_mask_dir (str): Directory to store validation masks.
    - test_img_dir (str): Directory to store test images.
    - test_mask_dir (str): Directory to store test masks.
    - train_size (float, optional): Proportion of data to be used for training.
    - test_size (float, optional): Proportion of data to be used for testing.

    This function:
    - Creates necessary directories for training, validation, and testing.
    - Splits the image and mask data into training, validation, and test sets.
    - Moves the images and masks to their respective directories.
    """
    create_dir(train_img_dir)
    create_dir(train_mask_dir)
    create_dir(val_img_dir)
    create_dir(val_mask_dir)
    create_dir(test_img_dir)
    create_dir(test_mask_dir)

    images = os.listdir(image_dir)
    masks = os.listdir(mask_dir)

    images.sort()
    masks.sort()

    # First split to separate out the test set
    train_images, val_test_images, train_masks, val_test_masks = tts(
        images,
        masks,
        test_size=0.3,
        random_state=42
    )

    # Second split to separate out the validation set from the remaining images
    val_images, test_images, val_masks, test_masks = tts(
        val_test_images,
        val_test_masks,
        test_size=0.5,
        random_state=42
    )

    # Create symbolic links for files in their respective directories
    def create_symlink(src, dst):
        if not os.path.exists(dst):  # Check if the symlink already exists
            os.symlink(src, dst)

    for img, msk in zip(train_images, train_masks):
        create_symlink(os.path.join(
            image_dir, img), os.path.join(train_img_dir, img)
                       )
        create_symlink(os.path.join(
            mask_dir, msk), os.path.join(train_mask_dir, msk)
                       )

    for img, msk in zip(val_images, val_masks):
        create_symlink(os.path.join(
            image_dir, img), os.path.join(val_img_dir, img)
                       )
        create_symlink(os.path.join(
            mask_dir, msk), os.path.join(val_mask_dir, msk)
                       )

    for img, msk in zip(test_images, test_masks):
        create_symlink(os.path.join(
            image_dir, img), os.path.join(test_img_dir, img)
                       )
        create_symlink(os.path.join(
            mask_dir, msk), os.path.join(test_mask_dir, msk)
                       )


def main():
    """
    Define main function.

    Set up directory paths for the dataset and calls the
    function to split the data into training, validation, and testing subsets.

    Directories:
    - image_dir: Directory containing the verified images.
    - mask_dir: Directory containing the corresponding labels/masks.
    - train_img_dir: Directory where training images will be saved.
    - train_mask_dir: Directory where training masks will be saved.
    - val_img_dir: Directory where validation images will be saved.
    - val_mask_dir: Directory where validation masks will be saved.
    - test_img_dir: Directory where testing images will be saved.
    - test_mask_dir: Directory where testing masks will be saved.
    """
    image_dir = (
        '/home/lhernandez2/Venus_SAR/Dataset/preprocessed_images'
        )
    mask_dir = (
        '/home/lhernandez2/Venus_SAR/Dataset/preprocessed_labels'
        )
    train_img_dir = (
        '/home/lhernandez2/Venus_SAR/Dataset/training_images'
        )
    train_mask_dir = (
        '/home/lhernandez2/Venus_SAR/Dataset/training_labels'
        )
    val_img_dir = (
        '/home/lhernandez2/Venus_SAR/Dataset/val_images'
        )
    val_mask_dir = (
        '/home/lhernandez2/Venus_SAR/Dataset/val_labels'
        )
    test_img_dir = (
        '/home/lhernandez2/Venus_SAR/Dataset/test_images'
        )
    test_mask_dir = (
        '/home/lhernandez2/Venus_SAR/Dataset/test_labels'
        )

    split_data(
        image_dir,
        mask_dir,
        train_img_dir,
        train_mask_dir,
        val_img_dir,
        val_mask_dir,
        test_img_dir,
        test_mask_dir
    )


if __name__ == '__main__':
    main()
