#from osgeo import gdal
import cv2
from matplotlib import pyplot as plt
import rasterio
import os
from tqdm import tqdm


def crop_image(image_path, x_offset, y_offset, x_size, y_size, output_path):
    # Open the image
    dataset = gdal.Open(image_path, gdal.GA_ReadOnly)

    # Check the number of bands
    num_bands = dataset.RasterCount

    # Read the chunk from the image using the specified offset and size
    chunk = dataset.ReadAsArray(x_offset, y_offset, x_size, y_size)

    # Create a new image to save the chunk
    driver = gdal.GetDriverByName('GTiff')
    output_path = 'D:\\Venus\\chunks\\chunk.tif'
    out_dataset = driver.Create(output_path, x_size, y_size, num_bands)

    # Copy the metadata from the original image
    out_dataset.SetGeoTransform(dataset.GetGeoTransform())
    out_dataset.SetProjection(dataset.GetProjection())

    # Write the chunk to the new image
    if num_bands == 1:
        out_dataset.GetRasterBand(1).WriteArray(chunk)
    else:
        for i in range(1, num_bands + 1):
            out_dataset.GetRasterBand(i).WriteArray(chunk[i - 1])

    # Close the datasets
    dataset = None
    out_dataset = None

    print(f'Cropped image saved as {output_path}')


def threshold_bright_pixels(image, threshold=122):
    # Convert the image to grayscale and apply a threshold
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresholded = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return thresholded


def plot_image(image, save_path=None):
    # save the image as png
    cv2.imwrite(save_path, image)

    # Plot the image using matplotlib and save it
    plt.figure(figsize=(16, 12))
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.title('Thresholded Image')
    plt.tight_layout()
    plt.close()


def convert2png(image_path, output_path):
    # Open the .tif image using rasterio and convert to png
    with rasterio.open(image_path) as src:
        img = src.read()
        # Save the image as a .png file
        cv2.imwrite(output_path, img[0])

    print(f'Image saved as {output_path}')


if __name__ == '__main__':
    """# Define the region to crop (x_offset, y_offset, x_size, y_size)
    # Example: Crop a 500x500 region starting from (1000, 1000)
    x_offset = 0 # Top left corner
    y_offset = 17000 # Top left corner
    x_size = 10000 # Width
    y_size = 10000 # Height
    
    # Crop the image
    crop_image('venus.tif', x_offset, y_offset, x_size, y_size, 'chunks\\chunk.tif')"""

    images = os.listdir('verified_pairs\\images')

    for img in tqdm(images, desc="Processing images"):
        # Load the image
        image_path = f'verified_pairs\\images\\{img}'
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        
        # Threshold the image to keep only the bright pixels
        thresholded = threshold_bright_pixels(image)
        
        # Plot the thresholded image
        plot_image(thresholded, f'verified_pairs\\HLOD_masks\\{img}')

    print('All images processed successfully')

    # Convert the .tif image to .png
    #convert2png('chunks\\chunk.tif', 'chunks\\chunk.png')