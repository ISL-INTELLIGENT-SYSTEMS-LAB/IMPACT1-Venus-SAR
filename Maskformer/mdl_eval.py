import os
import yaml
import torch
import numpy as np
from math import pi
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, roc_auc_score
from torch.nn import functional as F

import evaluate
from transformers import (
        MaskFormerImageProcessor,
        MaskFormerForInstanceSegmentation
        )

from MaskFormer_Training import TransformerDataSet


def save_precision_recall_curve(
        ground_truth,
        probabilities,
        save_path
        ):

    precision, recall, _ = precision_recall_curve(
            ground_truth.flatten(), 
            probabilities.flatten()
            )
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.', label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid()
    
    plt.savefig(save_path)
    plt.close()


def save_roc_curve(
        ground_truth,
        probabilities,
        save_path
        ):

    # Compute ROC curve
    fpr, tpr, _ = roc_curve(ground_truth.flatten(), probabilities.flatten())
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, marker='.', label='ROC curve')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')  # Random classifier line
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.grid()

    # Save the plot
    plt.savefig(save_path)
    plt.close()

    print(f"ROC curve saved at: {save_path}")


def save_segmentation_plot(
        segmentation_mask,
        ground_truth,
        original_image,
        save_path
        ):

    fig, axes = plt.subplots(1, 3, figsize=(15, 10))

    axes[0].imshow(segmentation_mask, cmap='gray')
    axes[0].set_title("Segmentation Mask")
    axes[0].axis('off')

    axes[1].imshow(ground_truth, cmap='gray')
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')

    axes[2].imshow(original_image)
    axes[2].set_title("Original Image")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def evaluate_model(
        data_dir,
        test_img_dir,
        test_seg_dir,
        input_filen_prefix,
        output_filen_prefix,
        img_ext,
        label_ext,
        mdl_path,
        results_save_dir
    ):
 
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

    device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
            )

    model = MaskFormerForInstanceSegmentation.from_pretrained(
        model_path,
        local_files_only=True
        ).to(device)

    processor = MaskFormerImageProcessor(
        do_reduce_labels=True,
        size=(352, 352),
        ignore_index=255,
        do_resize=False,
        do_rescale=True,
        do_normalize=True,
    )

    preds = []
    annotates = []
    probs = []

    for index in tqdm(range(len(test_dataset))):
        sample = test_dataset[index]
        sample_img = sample['image'].convert("RGB")
        target_size = sample_img.size[::-1]
        inputs = processor(
            images=sample_img,
            return_tensors="pt"
            ).to(device)

        model.eval()
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Convert logits to prob dist and resize
        mask_logits = outputs.masks_queries_logits
        mask_logits = mask_logits.squeeze(1)
        mask_logits_agg = mask_logits.mean(dim=1)
        mask_logits_resized = F.interpolate(
                mask_logits_agg.unsqueeze(1),
                size=(352, 352),
                mode='bilinear',
                align_corners=False
                ).squeeze(1)
        probabilities = F.sigmoid(mask_logits_resized)
        probs.append(probabilities)

        result = processor.post_process_semantic_segmentation(
            outputs=outputs,
            target_sizes=[target_size]
            )[0]
        sem_seg_msk = result.numpy()

        preds.append(sem_seg_msk)
        annotates.append(np.array(sample['annotation']))
        save_segmentation_plot(
                sem_seg_msk,
                sample['annotation'],
                sample_img,
                f'./train_val_results/plot-val-{index}.png'
                )
        print(f"Saved plot-{index}")


    metrics = evaluate.load("mean_iou")
    results = metrics.compute(
        references=annotates,
        predictions=preds,
        num_labels=2,
        ignore_index=255
        )

    print("mean_iou: ", round(results['mean_iou'] * 100, 2))
    print("mean_acc: ", round(results['mean_accuracy'] * 100, 2))
    print("overall_acc: ", round(results['overall_accuracy'] * 100, 2))
    print("per_class_iou: ", results['per_category_iou'])
    print("per_class_acc: ", results['per_category_accuracy'])
    auc_score = roc_auc_score(
            np.array(annotates).flatten(),
            np.array(probs).flatten()
            )
    print(f"AUC-ROC Score: {auc_score}")

    save_precision_recall_curve(
        np.array(annotates).flatten(),
        np.array(probs).flatten(),
        "./train_val_results/val-precision_recall_curve.png"
        )
    print("Saved precision-recall curve")

    save_roc_curve(
            np.array(annotates).flatten(),
            np.array(probs).flatten(),
            "./train_val_results/val-roc_curve.png"
            )
    print("Saved ROC curve")


if __name__ == "__main__":
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
    results_save_directory = model_path.replace(
            "weights",
            "results"
            )

    evaluate_model(
        data_directory,
        test_image_directory,
        test_segmentation_directory,
        input_filename_prefix,
        output_filename_prefix,
        image_extension,
        label_extension,
        model_path,
        results_save_directory
    )
