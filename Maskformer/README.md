
# Project README

This document describes the setup and execution of the following Python scripts:

1. move_images.py
2. preprocess_images.py
3. MaskFormer_Training.py
4. MaskFormer_Inference.py

## Prerequisites

Ensure you have the following installed:
- Python 3.9
- Required Python packages: `os`, `shutil`, `yaml`, `PIL`, `numpy`, `concurrent.futures`, `tqdm`, `torch`, `evaluate`, `sklearn`, `matplotlib`, `albumentations`, `filetype`, `transformers`

You can install the required packages using pip:
```bash
pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
```

You can download the current model files here along with the Ai4Mars dataset
```bash
git clone https://huggingface.co/facebook/maskformer-swin-base-ade <destination path>
```
[Ai4Mars Dataset](https://drive.google.com/file/d/1kUVkrdafKiyPMvH2z5NTFyUp9iSXqCX6/view?usp=sharing)

## Configuration Files

Each script requires a configuration file in YAML format. The required configuration files are:
- `move_images_config.yaml`
- `preprocess_images_config.yaml`
- `user_config_train.yaml`
- `user_config_test.yaml`

Ensure these files are present in the same directory as the scripts.

## Order of Operations

### Step 1: Move Images
The `move_images.py` script moves images and their corresponding labels to new directories based on the configuration file.

#### Usage:
```bash
python move_images.py
```

### Step 2: Preprocess Images
The `preprocess_images.py` script preprocesses the images and labels by performing operations such as resizing, splitting into quadrants, and saving the results.

#### Usage:
```bash
python preprocess_images.py
```

### Step 3: Train MaskFormer Model
The `MaskFormer_Training.py` script trains the MaskFormer model using the preprocessed images and labels. It saves the model and processor weights after training.

#### Usage:
```bash
python MaskFormer_Training.py
```

### Step 4: Perform Inference with MaskFormer
The `MaskFormer_Inference.py` script performs inference using the trained MaskFormer model and evaluates the results.

#### Usage:
```bash
python MaskFormer_Inference.py
```

## Detailed Script Descriptions

### move_images.py
This script moves images and their corresponding labels from source directories to destination directories specified in the `move_images_config.yaml` file.

### preprocess_images.py
This script preprocesses images and their labels. It reads images and labels from the directories specified in `preprocess_images_config.yaml`, processes them (e.g., splits into quadrants, resizes), and saves the processed files to output directories.

### MaskFormer_Training.py
This script trains the MaskFormer model on the dataset prepared by the previous scripts. It loads hyperparameters and paths from `user_config_train.yaml`, trains the model, and saves the model and processor weights to the specified location.

### MaskFormer_Inference.py
This script performs inference using the trained MaskFormer model. It loads the configuration from `user_config_test.yaml`, evaluates the model's performance, and saves the inference results and evaluation metrics.

## Notes

- Ensure all paths specified in the configuration files are correct.
- Adjust batch sizes and number of epochs in the configuration files based on your hardware capabilities and dataset size.
- For best performance, run the scripts on a machine with a compatible GPU.

