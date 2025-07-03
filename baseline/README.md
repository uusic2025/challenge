# Official Baseline for the UUSIC25 Challenge

[![Conference](https://img.shields.io/badge/MICCAI-2025-blue)](https://deep-breath-miccai.github.io/deepbreath-2025)
[![Challenge](https://img.shields.io/badge/Challenge-UUSIC25-brightgreen)](#)
[![arXiv](https://img.shields.io/badge/arXiv-UniUSNet-b31b1b.svg)]()

Welcome to the official baseline repository for the **Universal Ultrasound Image Challenge: Multi-Organ Classification and Segmentation (UUSIC25)**. This repository provides a complete pipeline to help you get started with the challenge.

The provided baseline is based on our previous work, [**UniUSNet: A Promptable Framework for Universal Ultrasound Disease Prediction and Tissue Segmentation**](https://github.com/Zehui-Lin/UniUSNet).

## 📖 Table of Contents

- [📖 Table of Contents](#-table-of-contents)
- [🎯 About the Challenge](#-about-the-challenge)
- [🚀 Getting Started](#-getting-started)
  - [1. Clone Repository](#1-clone-repository)
  - [2. Create Environment](#2-create-environment)
  - [3. Prepare Datasets](#3-prepare-datasets)
  - [4. Download Pre-trained Weights (Optional)](#4-download-pre-trained-weights-optional)
- [🏋️‍♀️ Model Training](#️-model-training)
- [🧪 Inference and Evaluation](#-inference-and-evaluation)
- [📦 Preparing Your Submission](#-preparing-your-submission)
- [📂 File Structure](#-file-structure)
- [❓ Frequently Asked Questions (FAQ)](#-frequently-asked-questions-faq)
- [©️ Citation](#️-citation)



## 💬 Got a Question or Found a Bug?

We highly encourage you to **[open an issue on our GitHub repository](https://github.com/uusic2025/challenge/issues)** for any questions or problems you encounter. This is the best way to get help, as it allows the entire community to benefit from the discussion and solutions.


## 🎯 About the Challenge

The UUSIC25 challenge aims to spur the development of universal models for ultrasound image analysis. Ultrasound imaging is a cornerstone of biomedical diagnostics, but creating models that perform accurate classification and segmentation across diverse organs and pathologies remains a significant hurdle.

This challenge encourages participants to develop innovative algorithms that can handle multiple tasks (classification and segmentation) on a wide range of ultrasound images from 7 different organs. Our goal is to push the boundaries of model generalization, efficiency, and clinical applicability.

This baseline provides a strong starting point, featuring a prompt-based multi-task architecture using a Swin Transformer backbone.



## 🚀 Getting Started

Follow these steps to set up your environment and run the baseline model.

### 1. Clone Repository

```bash
git clone https://github.com/uusic2025/challenge.git
cd challenge
```

### 2. Create Environment

We recommend using Conda to manage the environment.

```bash
# Create a new conda environment
conda create -n uusic25 python=3.10 -y
conda activate uusic25

# Install PyTorch (ensure compatibility with your CUDA version)
# Example for CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

A possible `requirements.txt` is provided below for your convenience:
```txt
numpy
opencv-python
yacs
timm
einops
scipy
tqdm
tensorboard
medpy
Pillow
```

### 3. Prepare Datasets

The data structure is crucial for the data loaders to work correctly.

1.  **Organize Data**: The data structure is crucial for the data loaders to work correctly, especially for aligning data with the one-hot encoders used in training. Please follow these steps carefully:
    1.  Inside the project folder, create a new directory named `data`.
    2.  **Unzip Public Data**: Unzip `Challenge_Data_Public.zip`. This will give you `classification` and `segmentation` folders. Place both of them inside your new `data` directory.
    3.  **Unzip Private Data**: Unzip `Challenge_Data_Private_v2_fully_anonymized.zip`. You will find `Train` and `Val` folders inside.
    4.  **Merge and Prefix Training Data**:
        -   Go into the private `Train` folder. It also contains `classification` and `segmentation` subfolders.
        -   For each dataset subfolder that came from this private `Train` data, you must **add the prefix `private_` to its name** (e.g., `Breast` becomes `private_Breast`, `Liver` becomes `private_Liver`). This step is critical for the code to work correctly.
        -   Merge these renamed private training folders with their public counterparts in `data/classification` and `data/segmentation`.
    5.  **Place Validation Data & JSON files**:
        -   Move the entire `Val` folder from the private zip directly into your `data` directory.
        -   Finally, place all `.json` ground truth files from both zips directly into the root of the `data` directory.

    Your final `data` directory structure should look similar to this:

    ```
    data/
    ├── classification
    │   ├── Appendix
    │   ├── BUS-BRA
    │   ... (other public datasets)
    │   ├── private_Appendix
    │   ├── private_Breast
    │   └── ... (other prefixed private datasets)
    ├── segmentation
    │   ├── BUS-BRA
    │   ... (other public datasets)
    │   ├── private_Breast
    │   └── ... (other prefixed private datasets)
    ├── Val
    │   ├── classification
    │   └── segmentation
    ├── private_test_for_participants.json
    ├── private_train_ground_truth.json
    ├── private_val_for_participants.json
    └── public_all_ground_truth.json
    ```

2.  **Generate File Lists**: After organizing the data, you need to generate `train.txt`, `val.txt`, and `test.txt` for each dataset. We provide a script to do this automatically.

    ```bash
    python datasets/generate_txt.py
    ```
    This script will scan the `data/` directory, split the files into 70% training, 20% validation, and 10% testing sets, and write the file paths into the corresponding `.txt` files.

### 4. Download Pre-trained Weights (Optional)

To get started quickly or if you want to **skip the training step** and jump directly to inference, you can use our provided baseline weights.

1.  **Download Model Weights**: Go to our official [**GitHub Releases page**](https://github.com/uusic2025/challenge/releases/latest) and download the `baseline.pth` file.
2.  **Prepare Experiment Directory**: Create a directory for your experiment output: `mkdir -p exp_out/trial_1`.
3.  **Place and Rename Weights**: Move the downloaded `baseline.pth` file into this folder and **rename it to `best_model.pth`**. The testing script `omni_test.py` is configured to load the model from `exp_out/trial_1/best_model.pth`.
4.  **(Recommended) Prepare Backbone Checkpoint**: The code may try to load a pre-trained Swin Transformer backbone. To prevent potential errors during initialization, create a `pretrained_ckpt` folder in the project root and place the `swin_tiny_patch4_window7_224.pth` file inside it. You can download this from the official Swin Transformer repository. This step is only for initializing the encoder's weights and is separate from loading our fine-tuned `best_model.pth`.

Now you can proceed directly to the [Inference and Evaluation](#-inference-and-evaluation) section using this model.

## 🏋️‍♀️ Model Training

The training process is handled by `omni_train.py`, which leverages `omni_trainer.py`. It uses a sophisticated weighted sampler to balance learning between different datasets and tasks (segmentation and classification).

To start training, run the provided shell script:

```bash
bash baseline.sh
```

Alternatively, you can run the command directly. For multi-GPU training (e.g., 2 GPUs):

```bash
python -m torch.distributed.launch \
    --use_env \
    --nproc_per_node=2 \
    --master_port=12345 \
    omni_train.py \
    --output_dir=exp_out/trial_1 \
    --prompt \
    --base_lr=0.003 \
    --batch_size=32 \
    --max_epochs=200
```

**Key Arguments**:
- `--output_dir`: Directory to save logs, checkpoints, and validation results.
- `--prompt`: Enables the prompt-based learning mechanism.
- `--batch_size`: Total batch size across all GPUs.
- `--max_epochs`: Total number of training epochs.
- `--pretrain_ckpt`: Path to a pretrained Swin Transformer checkpoint (`.pth`) to initialize the encoder. The baseline will automatically load from `pretrained_ckpt/swin_tiny_patch4_window7_224.pth`.

Checkpoints and logs will be saved in the specified `--output_dir`. The best-performing model on the validation set will be saved as `best_model.pth`.

## 🧪 Inference and Evaluation

After training, you can evaluate your model on the test sets using `omni_test.py`.

To run evaluation on a single GPU:

```bash
python -m torch.distributed.launch \
    --use_env \
    --nproc_per_node=1 \
    --master_port=12345 \
    omni_test.py \
    --output_dir=exp_out/trial_1 \
    --prompt \
    --is_saveout
```

**Key Arguments**:
- `--output_dir`: This should be the *same directory as your training output* (where `best_model.pth` is saved) or the directory where you placed the pre-trained weights.
- `--prompt`: Must be consistent with the training setting.
- `--is_saveout`: If specified, the script will save predicted masks and ground truths as images in `<output_dir>/predictions/`, which is useful for visual inspection.

Evaluation results (Dice for segmentation, Accuracy for classification) will be printed to the console and appended to `exp_out/result.csv`.

## 📦 Preparing Your Submission

The final submission will be a Docker container. The entry point for inference is `model.py`.

1.  **The `Model` Class**: `model.py` contains a `Model` class that initializes your network and loads your trained weights. The key method is `predict_segmentation_and_classification`.

2.  **Inference Logic**: This method iterates through a list of images, performs inference, and saves the results in the format required by the challenge.
    -   **Classification**: Results are compiled into a single `classification.json` file, mapping each image path to its predicted class and probabilities.
    -   **Segmentation**: Each predicted mask is saved as a PNG file, mirroring the input directory structure.

3.  **⚠️ Important Note on Segmentation Output**: The model performs inference on fixed-size images (e.g., 224x224). The baseline code in `model.py` already handles resizing the predicted masks back to the **original image dimensions** before saving. Please ensure your custom models do the same.

    ```python
    # From model.py - this is critical!
    resized_mask = cv2.resize(
        binary_mask_224, 
        original_size, 
        interpolation=cv2.INTER_NEAREST # Use nearest-neighbor to keep mask binary
    )
    ```

4.  **How to Test Your Submission Logic**: You can run `model.py` directly to simulate the evaluation process on the validation set.

    ```bash
    python model.py
    ```
    This script will:
    - Load the validation data list from `data/Val/private_val_for_participants.json`.
    - Use the trained model specified in `model.py` (ensure the path is correct).
    - Generate outputs in `exp_out/sample_result_submission/`.

You should wrap this logic in a Docker container according to the challenge submission guidelines.

## 📂 File Structure

<details>
<summary>Click to expand the project directory tree</summary>

```
.
├── baseline.sh
├── config.py
├── configs
│   └── swin_tiny_patch4_window7_224_lite.yaml
├── data
│   ├── classification
│   │   ├── Appendix
│   │   ├── BUS-BRA
│   │   ... (and other datasets)
│   └── segmentation
│       ├── BUS-BRA
│       │   ├── imgs
│       │   └── masks
│       ... (and other datasets)
├── datasets
│   ├── dataset.py
│   ├── generate_txt.py
│   └── omni_dataset.py
├── exp_out
│   ├── result.csv
│   └── trial_1
│       ├── best_model.pth
│       ├── log
│       └── log.txt
├── model.py
├── networks
│   └── omni_vision_transformer.py
├── omni_test.py
├── omni_trainer.py
├── omni_train.py
├── pretrained_ckpt
│   └── swin_tiny_patch4_window7_224.pth
├── README.md
├── requirements.txt
└── utils.py
```

</details>

- **`data/`**: Root directory for all datasets.
- **`configs/`**: Contains model configuration files (`.yaml`).
- **`datasets/`**: Data loading and processing scripts. `omni_dataset.py` is key for multi-task training.
- **`networks/`**: Model architecture, primarily `omni_vision_transformer.py`.
- **`exp_out/`**: Default output directory for experiments, logs, and checkpoints.
- **`model.py`**: The main script for generating submission files (your Docker entry point).
- **`omni_train.py` / `omni_test.py`**: Main scripts for training and testing.
- **`omni_trainer.py`**: Contains the core training and validation loops.
- **`baseline.sh`**: A convenience script to start training and testing.
- **`utils.py`**: Utility functions, including loss definitions and metrics.

## ❓ Frequently Asked Questions (FAQ)

**Q1: How do the "prompts" work in this model?**
**A:** The model uses prompt-based learning to guide its behavior. Instead of having separate models or heads for every single task, we feed it "prompts" as additional input vectors. The code defines four types of prompts:
- **`position_prompt`**: Tells the model which organ it's looking at (e.g., Breast, Kidney).
- **`task_prompt`**: Specifies the task, either `segmentation` or `classification`.
- **`nature_prompt`**: Informs the model if the target is an `organ` or a `tumor`.
- **`type_prompt`**: Describes the input view (e.g., `whole` image, `local` crop).
These prompts, defined in `omni_dataset.py` and used in `omni_vision_transformer.py`, allow a single model to flexibly handle our diverse multi-organ, multi-task challenge.

**Q2: Why are there two classification heads (`x_cls_2_way`, `x_cls_4_way`) in the model?**
**A:** This is a key feature for handling datasets with different numbers of classes within a single framework. Our challenge includes binary classification tasks (e.g., benign vs. malignant) and multi-class tasks (e.g., Luminal A/B, HER2, Triple-negative breast cancer, which has 4 classes). The model (`omni_vision_transformer.py`) has two separate linear heads, and the training logic in `omni_trainer.py` dynamically calculates the loss based on which type of data is in the current batch.

**Q3: How are the different datasets balanced during training?**
**A:** Datasets vary greatly in size. To prevent the model from being biased towards larger datasets, we use a `WeightedRandomSamplerDDP` (defined in `omni_dataset.py`). During training (`omni_trainer.py`), we assign higher sampling weights to smaller datasets, ensuring they are seen more frequently by the model. The specific weights are hard-coded in `omni_trainer.py`.

**Q4: How should I modify the code for my own custom model?**
**A:**
1.  **Network**: Replace the `OmniVisionTransformer` in `omni_train.py` and `model.py` with your own network class. Ensure your model's forward pass can handle the different tasks. A good approach is to return a tuple of outputs, similar to the baseline: `(segmentation_logits, classification_logits_2_way, classification_logits_4_way)`.
2.  **Data Loaders**: If your model requires different data augmentations, modify the `transform` in `omni_trainer.py`.
3.  **Submission**: The most important part is to update `model.py`. Make sure it correctly loads your custom model and weights and that its `predict_segmentation_and_classification` method works as expected.

**Q5: What happens if I encounter an organ not defined in the prompts?**
**A:** The code in `model.py` includes a fallback mechanism. If an organ name is not found in the `organ_to_position_map`, it defaults to the `'indis'` (indistinct) prompt. You may want to refine this for better generalization.
```python
# From model.py
position_key = organ_to_position_map.get(organ_name, 'indis')
```

## ©️ Citation

If you use this baseline or find our work helpful, please consider citing:

```bibtex
@inproceedings{lin2024uniusnet,
  title={UniUSNet: A Promptable Framework for Universal Ultrasound Disease Prediction and Tissue Segmentation},
  author={Lin, Zehui and Zhang, Zhuoneng and Hu, Xindi and Gao, Zhifan and Yang, Xin and Sun, Yue and Ni, Dong and Tan, Tao},
  booktitle={2024 IEEE International Conference on Bioinformatics and Biomedicine (BIBM)},
  pages={5550--5557},
  year={2024},
  organization={IEEE}
}

# Please also add a citation for the UUSIC25 challenge paper once it is available.
```
