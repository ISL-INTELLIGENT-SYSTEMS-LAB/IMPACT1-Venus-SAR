import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import os
from tqdm import tqdm


# Function to process each component
def process_component(label):
    component = (labels_im == label)
    if np.sum(component) > size_threshold:
        output_image[component] = 255

images = os.listdir('verified_pairs\\HLOD_masks')

# Define the size threshold to filter components
size_threshold = 75  # You can adjust this threshold

for img in tqdm(images, desc="Processing images"):
    # Load the image
    image_path = f"verified_pairs\\HLOD_masks\\{img}"
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Perform connected component analysis
    num_labels, labels_im = cv2.connectedComponents(image)

    # Create an output image to keep only large components
    output_image = np.zeros(image.shape, dtype=np.uint8)

    # Process components in parallel
    with ThreadPoolExecutor() as executor:
        executor.map(process_component, range(1, num_labels))  # Start from 1 to skip the background

    # Save the processed image
    processed_image_path = f'verified_pairs\\HLOD_masks\\{img}'
    cv2.imwrite(processed_image_path, output_image)

print('All images processed successfully')
