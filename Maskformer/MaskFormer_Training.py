import os
import torch
import matplotlib.pyplot as plt
import albumentations as A
import numpy as np
import yaml

from datetime import datetime
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
from torch.utils.data import Dataset, Subset, DataLoader
from transformers import (
    MaskFormerConfig,
    MaskFormerImageProcessor,
    MaskFormerModel,
    MaskFormerForInstanceSegmentation,
)

# Set environment variables for CUDA
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"  # Enable CUDA launch blocking
os.environ["TORCH_USE_CUDA_DSA"] = "1"  # Enable CUDA device-side assertions

print("Completed import libraries.")


class TransformerDataSet(Dataset):

    def __init__(
        self,
        root_directory,
        image_directory=None,
        segmentation_directory=None,
        image_fragment=None,
        segmentation_fragment=None,
        image_extension=".png",
        segmentation_extension=".png",
        dataset_limit=None
    ):
        images = []
        annotations = []
        file_names = []

        # Set default directories if none are provided
        if image_directory is None:
            image_directory = 'realimage'
        if segmentation_directory is None:
            segmentation_directory = 'seg'
        if image_fragment is None:
            image_fragment = 'realimage'
        if segmentation_fragment is None:
            segmentation_fragment = 'encoded_seg'

        image_path = os.path.join(root_directory, image_directory)
        segmentation_path = os.path.join(
            root_directory,
            segmentation_directory
        )

        files = sorted(os.listdir(image_path))

        # Limit the dataset if a limit is specified
        if dataset_limit is not None:
            files = files[:dataset_limit]

        # Load images and corresponding segmentation masks
        for file_name in tqdm(files, desc="Loading dataset"):
            file_path = os.path.join(image_path, file_name)
            if os.path.isfile(file_path) and file_path.lower().endswith((
                    'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'gif'
            )):
                segmentation_file_name = file_name.replace(
                    image_fragment,
                    segmentation_fragment
                ).replace(image_extension, segmentation_extension)
                segmentation_file_path = os.path.join(
                    segmentation_path,
                    segmentation_file_name
                )
                if os.path.exists(segmentation_file_path):
                    image = Image.open(file_path).convert('RGB')
                    images.append(image)
                    segmentation = Image.open(
                        segmentation_file_path
                    ).convert('L')
                    annotations.append(segmentation)
                    file_names.append(file_name)

        self.images = images
        self.annotations = annotations
        self.filenames = file_names

    def __getitem__(self, index):
        # Return the image, annotation, and filename for a given index
        sample = {
            "image": self.images[index],
            "annotation": self.annotations[index],
            "filename": self.filenames[index]
        }
        return sample

    def __len__(self):
        # Return the total number of samples
        return len(self.images)

    @staticmethod
    def train_val_dataset(
            dataset,
            validation_split=0.30,
            training_split=0.70,
            random_state=None,
            shuffle=True
    ):
        # Split the dataset into training and validation sets
        training_indices, validation_indices = train_test_split(
            list(range(len(dataset))),
            test_size=validation_split,
            train_size=training_split,
            shuffle=shuffle,
            random_state=random_state
        )
        datasets = {}
        datasets['train'] = Subset(dataset, training_indices)
        datasets['validation'] = Subset(dataset, validation_indices)
        return datasets


class ImageSegmentationDataset(Dataset):
    def __init__(
            self,
            dataset,
            image_processor,
            image_transformations=None,
            ignore_index=255
    ):
        self.dataset = dataset
        self.processor = image_processor
        self.transform = image_transformations
        self.ignore_index = ignore_index

    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.dataset)

    def __getitem__(self, index):
        # Get image and annotation arrays
        image_array = np.array(self.dataset[index]["image"].convert("RGB"))
        annotation_array = np.array(self.dataset[index]["annotation"])

        # Check if the shapes of the image and annotation match
        if annotation_array.shape != image_array.shape[:2]:
            raise ValueError(
                f"Mask and image shapes do not match at index {index}: ",
                f"{annotation_array.shape} vs {image_array.shape[:2]}"
            )

        # Apply transformations if provided
        if self.transform is not None:
            transformed = self.transform(
                image=image_array,
                mask=annotation_array
            )
            image_array = transformed["image"]
            annotation_array = transformed["mask"]
            image_array = image_array.transpose(2, 0, 1)

        # Map instance IDs to class labels
        instance_to_class_mapping = {}
        class_labels = np.unique(annotation_array)
        for label in class_labels:
            instance_ids = np.unique(annotation_array)
            instance_to_class_mapping.update({
                instance_id: label for instance_id in instance_ids
            })

        # Replace ignore index in annotations
        annotation_array[annotation_array == self.ignore_index] = 255

        # Process the image and annotation arrays
        inputs = self.processor(
            [image_array],
            [annotation_array],
            instance_id_to_semantic_id=instance_to_class_mapping,
            return_tensors="pt"
        )
        inputs = {
            key: value.squeeze() if isinstance(
                value, torch.Tensor
            ) else value[0] for key, value in inputs.items()}
        return inputs


print("Compiled Classes")


def collate_fn(batch_samples):
    # Stack the batch samples into a single batch
    pixel_values = torch.stack(
        [sample["pixel_values"] for sample in batch_samples]
    )
    pixel_mask = torch.stack(
        [sample["pixel_mask"] for sample in batch_samples]
    )
    mask_labels = [sample["mask_labels"] for sample in batch_samples]
    class_labels = [sample["class_labels"] for sample in batch_samples]
    return {
        "pixel_values": pixel_values,
        "pixel_mask": pixel_mask,
        "mask_labels": mask_labels,
        "class_labels": class_labels
    }


def create_loss_plots(
        train_loss_values,
        validation_loss_values,
        timestamp,
        save_directory
):
    # Plot training and validation loss over epochs
    fig, axs = plt.subplots(2, 1)

    axs[0].plot(list(range(len(train_loss_values))), train_loss_values)
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].set_title('Train Loss')

    axs[1].plot(list(range(
        len(validation_loss_values))), validation_loss_values
                )
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Loss')
    axs[1].set_title('Validation Loss')

    plt.subplots_adjust(hspace=0.5)
    plt.savefig(
        os.path.join(
            save_directory,
            f'Train_and_Val_loss_{timestamp}.png'),
        dpi=300
    )


def visualize_instance_seg_mask(
        segmentation_mask,
        label_to_color_mapping,
        mask_type='groundtruth'
):
    # Set the color mapping for ground truth masks
    if mask_type == 'groundtruth':
        label_to_color_mapping = {
            1: (0, 0, 0),
            0: (255, 255, 255),
        }

    # Prepare the segmentation mask for visualization
    segmentation_mask = np.where(
        segmentation_mask == -1, 255,
        segmentation_mask
    )
    image = np.zeros((
        segmentation_mask.shape[0],
        segmentation_mask.shape[1],
        3
    ), dtype=np.uint8)

    for height in range(image.shape[0]):
        for width in range(image.shape[1]):
            color_mask = segmentation_mask[height, width]
            if color_mask in label_to_color_mapping:
                image[height, width, :] = label_to_color_mapping[color_mask]
            else:
                print(
                    f'{color_mask} is not an index in the colormap. ',
                    'Using unknown color.')
                image[height, width, :] = (219, 52, 235)

    return image / 255.0


def get_inference(
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
    # Extract logits for class queries, excluding the "no object" class
    if hasattr(outputs, 'class_queries_logits'):
        # class_queries_logits shape: (batch_size, num_queries, num_classes)
        # For binary classification, focus on class 0 and class 1 only, ignoring class 2 (no object)
        print("ITWORKS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        class_logits = outputs.class_queries_logits.squeeze(0)  # Shape: (num_queries, num_classes)
        class_logits_filtered = class_logits[:, :2]  # Exclude the third class (no object)

        # Apply softmax to get probabilities for class 0 and class 1
        class_probs = torch.softmax(class_logits_filtered, dim=-1).cpu().numpy()

        # Extract probabilities for Class 0 and Class 1
        probs_class_0 = class_probs[:, 0]
        probs_class_1 = class_probs[:, 1]

        # Stack the probabilities to create confidence scores for both classes
        confidences = np.stack([probs_class_0, probs_class_1], axis=0)
    else:
        # Fallback if logits are not found
        print("Warning: class_queries_logits not found in outputs")
        confidences = np.ones(target_size)  # Fallback for confidence if no logits

    return semantic_segmentation_mask, confidences

def create_image_fromdataset(index, dataset, save_directory, timestamp):

    # Display and save an image and its annotation from the dataset
    print(
        "[INFO] Displaying an image from dataset index ",
        f"{index} and its annotation..."
    )

    image = dataset[index]["image"]
    image = np.array(image.convert("RGB"))
    annotation = dataset[index]["annotation"]
    annotation = np.array(annotation)

    plt.figure(figsize=(15, 5))
    for plot_index in range(2):
        if plot_index == 0:
            plot_image = image
            title = "Original"
        else:
            plot_image = annotation[..., plot_index - 1]
            title = ["Class Map (R)", "Instance Map (G)"][plot_index - 1]
        plt.subplot(1, 2, plot_index + 1)
        plt.imshow(plot_image)
        plt.title(title)
        plt.axis("off")
    plt.savefig(os.path.join(
        save_directory, f"Image_seg_{index}_{timestamp}.png"), dpi=300)


print("Compiled Methods")

# Load user configuration from YAML file
with open("user_config_train.yaml", "r") as file:
    usr_config = yaml.safe_load(file)

# Define configuration variables
check_preprocessed = False
EPOCHS = usr_config['EPOCHS']
NOW = datetime.now().strftime("%Y%m%d_%H%M%S")
ID2LABEL = usr_config['ID2LABEL']
LABEL2ID = usr_config['LABEL2ID']
label2color = usr_config['label2color']

PROJECTDIR = usr_config['PROJECTDIR']
subproject = usr_config['subproject']
DATADIR = usr_config['DATADIR']
SAVEROOT = usr_config['SAVEROOT']

IMGDIR_TRG = usr_config['IMGDIR_TRG']
IMGDIR_VAL = usr_config['IMGDIR_VAL']
IFRAG = usr_config['IFRAG']
SEGDIR_TRG = usr_config['SEGDIR_TRG']
SEGDIR_VAL = usr_config['SEGDIR_VAL']
SFRAG = usr_config['SFRAG']
IMFIX = usr_config['IMFIX']
LABFIX = usr_config['LABFIX']
IMAGE_SIZE_X = usr_config['IMAGE_SIZE_X']
IMAGE_SIZE_Y = usr_config['IMAGE_SIZE_Y']
THRESHOLD = usr_config['THRESHOLD']
RANDOM_SEED = usr_config['RANDOM_SEED']

MODEL_NAME = usr_config['MODEL_NAME']
LEARNING_RATE = float(usr_config['LEARNING_RATE'])
BATCH_SIZE = usr_config['BATCH_SIZE']
NUM_CLASSES = len(ID2LABEL)  # Ensure this matches the number of classes

ADE_MEAN = np.array(usr_config['ADE_MEAN']) / 255
ADE_STD = np.array(usr_config['ADE_STD']) / 255
ADE_MEAN = ADE_MEAN.tolist()
ADE_STD = ADE_STD.tolist()

LIMIT_DATASET = usr_config['LIMIT_DATASET']
print("Defined Constants")

if __name__ == "__main__":
    # Prepare directories for saving weights and results
    saveweights_location = os.path.join(SAVEROOT, f'weights_train_{NOW}')
    saveresults_location = os.path.join(SAVEROOT, f'results_train_{NOW}')

    os.makedirs(saveweights_location, exist_ok=True)
    os.makedirs(saveresults_location, exist_ok=True)

    train = TransformerDataSet(
        DATADIR,
        IMGDIR_TRG,
        SEGDIR_TRG,
        IFRAG,
        SFRAG,
        IMFIX,
        LABFIX,
        dataset_limit=LIMIT_DATASET
    )
    validation = TransformerDataSet(
        DATADIR,
        IMGDIR_VAL,
        SEGDIR_VAL,
        IFRAG,
        SFRAG,
        IMFIX,
        LABFIX,
        dataset_limit=LIMIT_DATASET
    )
    '''tf_dataset = TransformerDataSet(
        DATADIR,
        IMGDIR,
        SEGDIR,
        IFRAG,
        SFRAG,
        IMFIX,
        LABFIX,
        dataset_limit=LIMIT_DATASET
    )

    print(f"Length of dataset is {len(tf_dataset)}")
    print("Saved weights to: ", saveweights_location)
    print("Saved results to: ", saveresults_location)

    # Split the dataset into training and validation sets
    training_val_dataset = tf_dataset.train_val_dataset(
        tf_dataset,
        random_state=RANDOM_SEED
    )
    train = training_val_dataset["train"]
    validation = training_val_dataset["validation"]

    print(f"Length of training dataset is {len(train)}")
    print(f"Length of validation dataset is {len(validation)}")

    # Save file names for training and validation datasets
    train_files = [tf_dataset.filenames[idx] for idx in train.indices]
    val_files = [tf_dataset.filenames[idx] for idx in validation.indices]

    with open('files_split.txt', 'w') as f:
        f.write("Train files:\n")
        for file in train_files:
            f.write(file + '\n')
        f.write("\nVal files:\n")
        for file in val_files:
            f.write(file + '\n')'''

    print(f"Length of training dataset is {len(train)}")
    print(f"Length of validation dataset is {len(validation)}")

    # Create the MaskFormer Image Processor
    processor = MaskFormerImageProcessor(
        do_reduce_labels=True,
        size=(IMAGE_SIZE_X, IMAGE_SIZE_Y),
        ignore_index=255,
        do_resize=False,
        do_rescale=False,
        do_normalize=False,
    )

    print("Created the MaskFormer Image Processor")

    # Load the MaskFormer model and configuration
    local_model_dir = '/home/lhernandez2/Venus_SAR/Maskformer/Model/'
    config = MaskFormerConfig.from_pretrained(
        local_model_dir,
        num_labels=NUM_CLASSES,
        local_files_only=True
    )
    model = MaskFormerForInstanceSegmentation.from_pretrained(
        local_model_dir,
        config=config,
        local_files_only=True,
        ignore_mismatched_sizes=True
    )

    config.id2label = ID2LABEL
    config.label2id = LABEL2ID
    print(config)

    # Load the base MaskFormer model
    base_model = MaskFormerModel.from_pretrained(
        local_model_dir,
        local_files_only=True
    )
    model.model = base_model

    # Load user configuration again
    with open("user_config_train.yaml", "r") as file:
        usr_config = yaml.safe_load(file)

    # Define augmentation transformations
    train_transform = A.Compose([
        A.Resize(width=IMAGE_SIZE_X, height=IMAGE_SIZE_Y),
        A.HorizontalFlip(p=0.3),
        A.Normalize(mean=ADE_MEAN, std=ADE_STD),
    ])

    val_transform = A.Compose([
        A.Resize(width=IMAGE_SIZE_X, height=IMAGE_SIZE_Y),
        A.Normalize(mean=ADE_MEAN, std=ADE_STD),
    ])

    print("Created the augmentation transforms...")

    # Create the train and validation datasets for instance segmentation
    train_dataset = ImageSegmentationDataset(
        train,
        image_processor=processor,
        image_transformations=train_transform
    )
    val_dataset = ImageSegmentationDataset(
        validation,
        image_processor=processor,
        image_transformations=train_transform
    )

    print("Created the train and validation instance segmentation dataset...")

    if check_preprocessed:
        # Print the first validation file and its preprocessed inputs
        print(validation[0]['filename'])
        inputs = val_dataset[0]
        for key, value in inputs.items():
            print("[INFO] Displaying a shape of the preprocessed inputs...")
            print(key, value.shape)

        for key, value in inputs.items():
            print("[INFO] Displaying arrays of the preprocessed inputs...")
            print(key, value)

    print("Building the training and validation dataloader...")

    # Adjust the batch size if necessary
    BATCH_SIZE = min(BATCH_SIZE, 2)  # Reduce batch size to 2 or the value in
    # the configuration, whichever is smaller

    # Create the DataLoader for training and validation datasets
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    num_epochs = EPOCHS
    avg_train = []
    avg_val = []
    best_loss = 500
    best_epoch = 0

    # Check if CUDA is available and set the device
    print(f'Is cuda available? {torch.cuda.is_available()}')
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )  # Switch back to GPU
    model.to(device)

    # Initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training and validation loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs} | Training")
        model.train()
        train_loss, val_loss = [], []

        # Training loop
        with tqdm(
                train_dataloader,
                desc=f"Training Epoch {epoch+1}/{num_epochs}"
                  ) as train_pbar:
            for idx, batch in enumerate(train_pbar):
                optimizer.zero_grad()

                try:
                    # Forward pass and compute loss
                    outputs = model(
                        pixel_values=batch["pixel_values"].to(device),
                        mask_labels=[
                            labels.to(device) for labels in batch[
                                "mask_labels"]],
                        class_labels=[
                            labels.to(device) for labels in batch[
                                "class_labels"]],
                    )
                    loss = outputs.loss

                    # Backward pass and optimizer step
                    loss.backward()
                    optimizer.step()
                    train_loss.append(loss.item())
                    if idx % 50 == 0:
                        train_pbar.set_postfix({
                            "Training loss": round(
                                sum(train_loss) / len(train_loss), 6)})

                except RuntimeError as e:
                    print(f"RuntimeError at batch {idx}: {e}")
                    for key, value in batch.items():
                        if isinstance(value, torch.Tensor):
                            print(
                                f"Batch key: {key}, ",
                                f"shape: {value.shape} ",
                                f"device: {value.device}"
                            )
                        else:
                            print(f"Batch key: {key}, length: {len(value)}")
                    raise

        # Compute average training loss for the epoch
        avg_train_loss = sum(train_loss) / len(train_loss)
        avg_train.append(avg_train_loss)

        model.eval()
        print(f"Epoch {epoch+1}/{num_epochs} | Validation")

        # Validation loop
        with tqdm(
                val_dataloader,
                desc=f"Validation Epoch {epoch+1}/{num_epochs}"
        ) as val_pbar:
            for idx, batch in enumerate(val_pbar):
                try:
                    # Forward pass and compute loss for validation
                    with torch.no_grad():
                        outputs = model(
                            pixel_values=batch["pixel_values"].to(device),
                            mask_labels=[
                                labels.to(device) for labels in batch[
                                    "mask_labels"]],
                            class_labels=[
                                labels.to(device) for labels in batch[
                                    "class_labels"]],
                        )
                        loss = outputs.loss
                        if loss.item() < best_loss:
                            best_loss = loss.item()
                            model.save_pretrained(saveweights_location)
                            print(f"Best loss {best_loss}; mdl saved")
                            best_epoch = epoch + 1

                        val_loss.append(loss.item())
                        if idx % 50 == 0:
                            val_pbar.set_postfix({
                                "Validation loss": round(
                                    sum(val_loss) / len(val_loss), 6)})

                except RuntimeError as e:
                    print(f"RuntimeError at validation batch {idx}: {e}")
                    for key, value in batch.items():
                        if isinstance(value, torch.Tensor):
                            print(
                                f"Validation batch key: {key}, "
                                f"shape: {value.shape}, "
                                f"device: {value.device}"
                            )
                        else:
                            print(
                                f"Validation batch key: {key}, ",
                                f"length: {len(value)}")
                    raise

        # Compute average validation loss for the epoch
        avg_val_loss = sum(val_loss) / len(val_loss)
        avg_val.append(avg_val_loss)
        print(
            f"Epoch {epoch+1}/{num_epochs} ",
            f"| train_loss: {avg_train_loss} ",
            f"| validation_loss: {avg_val_loss}"
        )
        if loss.item() < best_loss:
            best_loss = loss.item()
            model.save_pretrained(saveweights_location)
            print(f"Best loss {best_loss}; mdl saved")
            best_epoch = epoch + 1

    # Plot and save the training and validation loss curves
    create_loss_plots(avg_train, avg_val, NOW, saveresults_location)

    # Save the trained model
    # model.save_pretrained(saveweights_location)
    # ^^^ was NOT saving best performing 

    # Update the processor configuration and save it
    processor.do_normalize = True
    processor.do_resize = True
    processor.do_rescale = True

    processor.save_pretrained(saveweights_location)

    print(f"\n**Best checkpoint at epoch {best_epoch} with loss {best_loss}**\n")
