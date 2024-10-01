# Import necessary libraries and modules
import os
import yaml
import torch
from math import pi
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, jaccard_score
)
import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm
from transformers import (
    MaskFormerImageProcessor,
    MaskFormerForInstanceSegmentation,
)
from MaskFormer_Training import TransformerDataSet
import evaluate

def calculate_metrics(y_true, y_pred, num_classes=2):
    """
    Calculate metrics for semantic segmentation.

    Parameters:
    - y_true: Ground truth labels (flattened 1D array or 2D matrix).
    - y_pred: Predicted labels (flattened 1D array or 2D matrix).
    - num_classes: Number of classes in segmentation task
    (default is 2 for binary segmentation).

    Returns:
    A dictionary of calculated metrics.
    """
    # Flatten if the input is a 2D matrix (like an image)
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    #Check if input have predictions for both classes
    unique_classes_true = np.unique(y_true)
    unique_class_pred = np.unique(y_pred)

    metrics = {}

    # 1. Overall Accuracy
    overall_accuracy = accuracy_score(y_true, y_pred)
    metrics['overall_accuracy'] = overall_accuracy

    # 2. Average Accuracy
    per_class_accuracies = []
    for cls in range(num_classes):
        cls_idx = (y_true == cls)
        if np.sum(cls_idx) == 0:
            continue
        class_accuracy = np.sum(
            y_pred[cls_idx] == y_true[cls_idx]
        ) / np.sum(cls_idx)
        per_class_accuracies.append(class_accuracy)

    avg_accuracy = np.mean(per_class_accuracies)
    metrics['average_accuracy'] = avg_accuracy

    # 3. Precision per class
    precision_per_class = precision_score(
        y_true, y_pred, average=None, labels=np.arange(
            num_classes
        ), zero_division=0
    )
    metrics['precision_per_class'] = precision_per_class

    # 4. Recall per class
    recall_per_class = recall_score(
        y_true, y_pred, average=None, labels=np.arange(
            num_classes
        ), zero_division=0
    )
    metrics['recall_per_class'] = recall_per_class

    # 5. F1 Score per class
    f1_per_class = f1_score(
        y_true, y_pred, average=None, labels=np.arange(
            num_classes
        ), zero_division=0
    )
    metrics['f1_score_per_class'] = f1_per_class

    # 6. IoU (Jaccard Index) per class
    iou_per_class = jaccard_score(
        y_true, y_pred, average=None, labels=np.arange(
            num_classes
        ), zero_division=0
    )
    metrics['iou_per_class'] = iou_per_class

    return metrics


def print_metrics(metrics, num_classes=2):
    """
    Print the calculated metrics in a readable format.

    Parameters:
    - metrics: Dictionary containing all calculated metrics.
    - num_classes: Number of classes in segmentation task
    (default is 2 for binary segmentation).
    """
    print(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}")
    print(f"Average Accuracy: {metrics['average_accuracy']:.4f}")

    for cls in range(num_classes):
        print(f"\nClass {cls}:")
        print(f"  Precision: {metrics['precision_per_class'][cls]:.4f}")
        print(f"  Recall: {metrics['recall_per_class'][cls]:.4f}")
        print(f"  F1 Score: {metrics['f1_score_per_class'][cls]:.4f}")
        print(f"  IoU: {metrics['iou_per_class'][cls]:.4f}")


def plot_pr_curve(
        y_true,
        y_pred_proba,
        num_classes=2,
        results_save_directory="results"
):
    y_true_bin = label_binarize(y_true, classes=[0, 1])

    plt.figure(figsize=(8, 6))

    for i in range(num_classes):
        precision, recall, _ = precision_recall_curve(
            y_true_bin[:, i],
            y_pred_proba[:, i]
        )
        ap_score = average_precision_score(
            y_true_bin[:, i], y_pred_proba[:, i]
        )

        # Plot PR curve
        plt.plot(
            recall, precision, label=f'Class {i} (AP = {ap_score:.2f})'
        )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for Each Class')
    plt.legend(loc='lower left')

    # Save the PR plot
    plt.savefig(
        os.path.join(results_save_directory, "pr_curve.jpg"
                     ), dpi=400, bbox_inches='tight'
    )
    plt.show()

def get_outputs(
        dataset_sample,
        processor,
        device,
        model,
        image_extension,
        threshold=None
):
    # Convert image to RGB and prepare for inference
    image = dataset_sample["image"].convert("RGB")
    target_size = image.size[::-1]  # Get the target size in (height, width)

    inputs = processor(images=image, return_tensors="pt").to(device)
    print("Preprocessed input:::::: ", inputs["pixel_values"].dtype)
    # Run inference with the model
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process the segmentation using semantic segmentation post-processing
    result = processor.post_process_semantic_segmentation(
        outputs=outputs,
        target_sizes=[target_size]
    )[0]  # Get the segmentation map for the current sample

    # Convert the result to a NumPy array (semantic segmentation map)
    semantic_segmentation_mask = result.cpu().numpy()

    return semantic_segmentation_mask, outputs

def main():
    # Load configuration file
    with open("user_config_test.yaml", "r") as file:
        config = yaml.safe_load(file)

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
    test_dataset = TransformerDataSet(
        data_directory,
        test_image_directory,
        test_segmentation_directory,
        input_filename_prefix,
        output_filename_prefix,
        image_extension,
        label_extension
    )
    print(f"Test dataset size: {len(test_dataset)}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MaskFormerForInstanceSegmentation.from_pretrained(
        model_path, local_files_only=True
    ).to(device)

    processor = MaskFormerImageProcessor(
        do_reduce_labels=True,
        size=(352, 352),
        ignore_index=255,
        do_resize=False,
        do_rescale=True,
        do_normalize=True,
    )
   

    # Randomly select indices for display
    display_indices = list(range(len(test_dataset)))
    print(f"Selected indices for display: {display_indices}")

    ground_truth_annotations = []
    predicted_masks = []
    total_pixel_count = 0
    confusion_matrix_sum = np.zeros((2, 2))

    # Initialize variables for confidence matrix
    predictions = []
    annotations = []
    all_confidences = []
    accuracy_values = []
    precision_values = []
    f1_values = []
    mIOU_values = []

    for index in tqdm(range(len(test_dataset))):
        test_sample = test_dataset[index]
        original_filename = test_sample['filename']

        # Get inference
        predicted_segmentation_mask, confidences = get_outputs(
            test_sample, processor, device, model, image_extension
        )
        if predicted_segmentation_mask is None:
            print(f"Warning: No prediction for {original_filename}")
            continue

        predictions.append(list(predicted_segmentation_mask))
        annotations.append(np.array(test_sample['annotation']))

        continue
         

        binary_predicted_mask = predicted_segmentation_mask.copy()
        binary_predicted_mask[binary_predicted_mask != 1] = 0
        binary_predicted_mask = binary_predicted_mask.astype("int32")

        annotation_array = np.array(test_sample["annotation"])
        if annotation_array.ndim == 3:
            ground_truth_mask = annotation_array[:, :, 0]
        else:
            ground_truth_mask = np.expand_dims(
                annotation_array, axis=-1
            )[:, :, 0]

        binary_ground_truth_mask = ground_truth_mask.copy()
        binary_ground_truth_mask[binary_ground_truth_mask == 255] = 0
        binary_ground_truth_mask = binary_ground_truth_mask.astype("int32")
        ground_truth_mask = ground_truth_mask.astype("int32")

        ground_truth_annotations.append(ground_truth_mask)
        predicted_masks.append(predicted_segmentation_mask)

        # Compute confusion matrix
        cm_true = confusion_matrix(
            binary_predicted_mask.flatten(),
            binary_ground_truth_mask.flatten(),
            labels=list(range(2)),
            normalize='true'
        )
        cm_pred = confusion_matrix(
            binary_predicted_mask.flatten(),
            binary_ground_truth_mask.flatten(),
            labels=list(range(2)),
            normalize='pred'
        )
        cm_all = confusion_matrix(
            binary_predicted_mask.flatten(),
            binary_ground_truth_mask.flatten(),
            labels=list(range(2)),
            normalize='all'
        )
        confusion_matrix_sum += cm_true
        total_pixel_count += len(binary_ground_truth_mask.flatten())

        # Store predictions, ground truths, and confidences
        all_predictions.extend(binary_predicted_mask.flatten())
        all_ground_truths.extend(binary_ground_truth_mask.flatten())
        all_confidences.extend(confidences.flatten())

    metrics = evaluate.load("mean_iou")

    results = metrics.compute(
        references=annotations,
        predictions=predictions,
        num_labels=2,
        ignore_index=None
        )

    print(results)
 

   # Calculate metrics using your custom metrics function
    metrics = calculate_metrics(
        np.array(all_ground_truths), np.array(all_predictions), num_classes=2
    )

    # Print the calculated metrics
    print_metrics(metrics, num_classes=2)

    # Visualize the confusion matrix
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_true,
        display_labels=["Background", "Wrinkle_Ridge"]
        )
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - Norm over true (rows) \n\n")
    plt.savefig(os.path.join(
        results_save_directory, "confusion_matrix_trueNorm_results.jpg"
    ), dpi=400, bbox_inches='tight'
                )
    plt.close()

    # Visualize the confusion matrix
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_pred,
        display_labels=['Background', 'Wrinkle_Ridge']
        )
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix - Norm over pred (columns)\n\n")
    plt.savefig(os.path.join(
        results_save_directory, "confusion_matrix_predNorm_results.jpg"
    ), dpi=400, bbox_inches="tight"
                )
    plt.close()

    # Visualize the confusion matrix
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_all,
        display_labels=["Background", "Wrinkle_ridge"]
        )
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix for All the Population\n\n")
    plt.savefig(os.path.join(
        results_save_directory, "confusion_matrix_allNorm_results.jpg"
    ), dpi=400, bbox_inches='tight'
                )
    plt.close()

    classes = ['Background', 'Wrinkle_Ridge']
    num_vars = 4
    angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    for idx, class_name in enumerate(classes):
        values = [
            metrics['precision_per_class'][idx],
            metrics['recall_per_class'][idx],
            metrics['iou_per_class'][idx],
            metrics['f1_score_per_class'][idx]
        ]
        values += values[:1]

        ax.plot(angles, values, linewidth=2, label=class_name)
        ax.fill(angles, values, alpha=0.25)

        for i, value in enumerate(values[:-1]):
            angle_rad = angles[i]
            ax.text(
                angle_rad,
                value - 0.05,
                f'{value:.2f}',
                horizontalalignment='center',
                size=8,
                color='black'
            )

    # Define metric labels for radar plot
    metric_labels = ['Precision', 'Recall', 'IoU', 'F1']

    # Set the labels for each angle on the radar chart
    plt.xticks(angles[:-1], metric_labels)
    ax.tick_params(pad=20)

    # Add legend and title
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.title('Radar Plot of Metrics per Class')

    # Save and show the radar plot
    plt.savefig(
        os.path.join(
            results_save_directory, "radar_chart.jpg"
        ), dpi=400, bbox_inches='tight'
    )
    plt.show()

    y_pred_proba = np.array(all_confidences).reshape(-1, 2)
    plt.hist(y_pred_proba[:, 0], bins=50, alpha=0.5, label='Class 0')
    plt.hist(y_pred_proba[:, 1], bins=50, alpha=0.5, label='Class 1')
    plt.legend()
    plt.title("Distribution of Predicted Probabilities")
    plt.show()

    plot_pr_curve(
        np.array(all_ground_truths).reshape(-1, 2),
        np.array(all_confidences).reshape(-1, 2),
        num_classes=2,
        results_save_directory=results_save_directory
    )
    print (y_true.shape)
    print (y_pred_proba.shape)

if __name__ == "__main__":
    main()
