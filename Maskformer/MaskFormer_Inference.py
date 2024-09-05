#################################################################################################
#
#   Fayetteville State University Intelligence Systems Laboratory (FSU-ISL)
#
#   Mentors:
#           Dr. Sambit Bhattacharya
#
#   File Name:
#          MaskFormer_Inference.py
#
#   Programmers:
#           Catherine Spooner
#           Carley Brinkley
#           Taylor Brown
#           Charlene Brighter
#
#################################################################################################

# Import necessary libraries and modules
import os
import yaml
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
from transformers import (
    MaskFormerImageProcessor,
    MaskFormerForInstanceSegmentation,
)
from MaskFormer_Training import *

# Function to load configuration from a YAML file
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

# Function to calculate overall accuracy
def calculate_overall_accuracy(true_positive, total_pixel_count):
    return np.sum(true_positive) / total_pixel_count

# Function to calculate mean accuracy
def calculate_mean_accuracy(true_positive, false_negative):
    class_accuracies = true_positive / (true_positive + false_negative)
    return np.nanmean(class_accuracies)

# Function to calculate mean IoU
def calculate_mean_iou(true_positive, false_positive, false_negative):
    iou = true_positive / (true_positive + false_positive + false_negative)
    return np.nanmean(iou)

# Function to compute confidence matrix
def compute_confidence_matrix(predictions, ground_truths, confidences, num_classes):
    confidence_matrix = np.zeros((num_classes, num_classes))
    for p, gt, conf in zip(predictions, ground_truths, confidences):
        confidence_matrix[gt, p] += conf
    # Normalize the matrix by the sum of confidences for each ground truth class
    for gt in range(num_classes):
        sum_confidences = np.sum(confidence_matrix[gt])
        if sum_confidences > 0:
            confidence_matrix[gt] /= sum_confidences
    return confidence_matrix

def main():
    # Load configuration file
    config = load_config('/home/lhernandez2/Venus_SAR/Maskformer/user_config_test.yaml')
    
    data_directory = config['DATADIR']
    test_image_directory = config['TEST_IMGDIR']
    test_segmentation_directory = config['TEST_SEGDIR']
    input_filename_prefix = config['IFRAG']
    output_filename_prefix = config['SFRAG']
    image_extension = config['IMFIX']
    label_extension = config['LABFIX']
    model_path = config['MODEL_LOCATION']
    results_save_directory = model_path.replace("weights", "results")
    
    # Create test dataset
    print("Creating test dataset...")
    test_dataset = TransformerDataSet(data_directory, test_image_directory, test_segmentation_directory,
                                      input_filename_prefix, output_filename_prefix, image_extension, label_extension)
    print(f"Test dataset size: {len(test_dataset)}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MaskFormerForInstanceSegmentation.from_pretrained(model_path, local_files_only=True).to(device)
    processor = MaskFormerImageProcessor.from_pretrained(model_path, local_files_only=True)

    # Randomly select indices for display
    display_indices = list(range(len(test_dataset)))
    print(f"Selected indices for display: {display_indices}")

    ground_truth_annotations = []
    predicted_masks = []
    total_pixel_count = 0
    confusion_matrix_sum = np.zeros((2, 2))

    # Initialize variables for confidence matrix
    all_predictions = []
    all_ground_truths = []
    all_confidences = []

    for index in tqdm(range(len(test_dataset))):
        test_sample = test_dataset[index]
        original_filename = test_sample['filename']
        result_filename = original_filename.replace(image_extension, "")

        # Get inference
        predicted_segmentation_mask, confidences = get_inference(test_sample, processor, device, model, image_extension)
        if predicted_segmentation_mask is None:
            print(f"Warning: No prediction for {original_filename}")
            continue
        
        binary_predicted_mask = predicted_segmentation_mask.copy()
        binary_predicted_mask[binary_predicted_mask == 0] = 1
        binary_predicted_mask[binary_predicted_mask == -1] = 0
        binary_predicted_mask = binary_predicted_mask.astype("int32")

        predicted_segmentation_mask[predicted_segmentation_mask == -1] = 255
        predicted_segmentation_mask = predicted_segmentation_mask.astype("int32")

        annotation_array = np.array(test_sample["annotation"])
        if annotation_array.ndim == 3:
            ground_truth_mask = annotation_array[:, :, 0]
        else:
            ground_truth_mask = np.expand_dims(annotation_array, axis=-1)[:, :, 0]
            
        binary_ground_truth_mask = ground_truth_mask.copy()
        binary_ground_truth_mask[binary_ground_truth_mask == 255] = 0
        binary_ground_truth_mask = binary_ground_truth_mask.astype("int32")
        ground_truth_mask = ground_truth_mask.astype("int32")

        ground_truth_annotations.append(ground_truth_mask)
        predicted_masks.append(predicted_segmentation_mask)

        # Compute confusion matrix
        cm = confusion_matrix(binary_predicted_mask.flatten(), binary_ground_truth_mask.flatten(), labels=list(range(2)))
        confusion_matrix_sum += cm
        total_pixel_count += len(binary_ground_truth_mask.flatten())

        # Store predictions, ground truths, and confidences
        all_predictions.extend(binary_predicted_mask.flatten())
        all_ground_truths.extend(binary_ground_truth_mask.flatten())
        all_confidences.extend(confidences.flatten())

        if index in display_indices:
            if np.any(predicted_segmentation_mask != 255):
                segmented_mask_display = visualize_instance_seg_mask(predicted_segmentation_mask, label2color, "semseg")
            else:
                segmented_mask_display = np.zeros((predicted_segmentation_mask.shape[0], predicted_segmentation_mask.shape[1], 3), dtype=np.uint8)

            ground_truth_display = visualize_instance_seg_mask(ground_truth_mask, label2color, "groundtruth")

            plt.figure(figsize=(10, 10))
            plt.style.use('_mpl-gallery-nogrid')
            plt.subplot(1, 3, 1)
            plt.title("Original Image")
            plt.imshow(test_sample["image"].convert("RGB"))
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.title("Ground Truth Annotation")
            plt.imshow(ground_truth_display)
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.title("Predicted Segmentation")
            plt.imshow(segmented_mask_display)
            plt.axis("off")

            save_path = os.path.join(results_save_directory, f"results_{result_filename}.jpg")
            #print(f"Saving image to {save_path}")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

    # Extract values from the confusion matrix
    true_positive = np.diag(confusion_matrix_sum)
    false_positive = np.sum(confusion_matrix_sum, axis=0) - true_positive
    false_negative = np.sum(confusion_matrix_sum, axis=1) - true_positive

    # Calculate metrics
    overall_accuracy = calculate_overall_accuracy(true_positive, total_pixel_count)
    mean_accuracy = calculate_mean_accuracy(true_positive, false_negative)
    mean_iou = calculate_mean_iou(true_positive, false_positive, false_negative)

    print(f"Mean IoU: {mean_iou} | Mean Accuracy: {mean_accuracy} | Overall Accuracy: {overall_accuracy}")

    # Visualize the confusion matrix
    normalized_confusion_matrix = confusion_matrix_sum / total_pixel_count
    disp = ConfusionMatrixDisplay(confusion_matrix=normalized_confusion_matrix)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix for Semantic Segmentation\n\n")
    plt.savefig(os.path.join(results_save_directory, "confusion_matrix_results.jpg"), dpi=400, bbox_inches='tight')
    plt.close()

    # Compute and visualize the confidence matrix
    num_classes = 2  # Assuming binary classification
    confidence_matrix = compute_confidence_matrix(np.array(all_predictions), np.array(all_ground_truths), np.array(all_confidences), num_classes)
    plt.figure(figsize=(10, 7))
    plt.imshow(confidence_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confidence Matrix")
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.savefig(os.path.join(results_save_directory, "confidence_matrix_results.jpg"), dpi=400, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()
