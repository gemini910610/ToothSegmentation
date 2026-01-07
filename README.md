# Periodontal Bone Loss Evaluation using Deep Learning-Based Image Segmentation

## 📦 Project Structure

```
📁 scripts
├── 📂 tools
│   └── 📄 visualize.py     # Visualizes model predictions alongside ground-truth masks for qualitative analysis
├── 📄 compare.py           # Compares predictions with ground-truth masks
├── 📄 dcm2png.py           # Converts DICOM series into PNG slices
├── 📄 download.py          # Downloads experiment logs from a remote server via SFTP
├── 📄 evaluate.py          # Runs model evaluation for each fold, computes losses and metrics (e.g., mIoU)
├── 📄 inference.py         # Generates segmentation masks by inference
├── 📄 prepare_kfold.py     # Data splitting for K-Fold cross-validation
├── 📄 run_experiment.py    # Main script to run complete experimental workflows
└── 📄 train.py             # Entry point for single model training
📁 src
├── 📂 models
│   ├── 📄 unet.py          # U-Net model architecture definition
│   └── 📄 ...              # Additional model implementation
├── 📄 config.py            # Configuration settings and hyperparameters
├── 📄 console.py           # Console output helpers (progress tracking and table-style summaries)
├── 📄 dataset.py           # Custom Dataset and DataLoader implementation
├── 📄 downloader.py        # Utilities for downloading experiment directories from a remote server using SFTP
├── 📄 losses.py            # Custom loss functions
├── 📄 metrics.py           # Evaluation metrics (e.g., mIoU)
├── 📄 optimizers.py        # Optimizer construction utilities
├── 📄 summary.py           # TensorBoard log parsing and scalar summary utilities
└── 📄 trainer.py           # Handles training and validation loops
```

## 📁 Dataset Preparation

The dataset must follow the directory structure below:
```
📁 datasets/<DATASET_NAME>
├── 📂 image
│   ├── 📂 data_1
│   │   ├── 📄 91.png
│   │   └── 📄 ...
│   └── 📂 ...
└── 📂 mask
    ├── 📂 data_1
    │   ├── 📄 91.png
    │   └── 📄 ...
    └── 📂 ...
```
**Requirements**
* Images and masks must share identical folder/file names.
* Mask images must contain pixel labels `{0, 1, 2}` corresponding 3 classes.

### DICOM to PNG Conversion

You can use the following script to convert DICOM series into PNG slices:
```
python -m scripts.dcm2png <DICOM_DIR> datasets/<DATASET_NAME>/image
```
This command converts each DICOM series into slice-wise PNG images and saves them under:
```
datasets/<DATASET_NAME>/image/data_<ID>/*.png
```

### Inference (Generate Masks)

You can use a trained model checkpoint to perform inference-only segmentation:
```
python -m scripts.inference <EXPERIMENT_NAME> datasets/<DATASET_NAME>/image datasets/<DATASET_NAME>/mask --fold <FOLD>
```
Notes
* `--fold <FOLD>` is used to select the trained model checkpoint from logs/<EXPERIMENT_NAME>/Fold_<FOLD>/best.pth

## ⚙️ Configuration (`configs/config.toml`)

```toml=
# System and Experiment
experiment = "UNet_baseline"
seed = 42
num_workers = 4

# Data Configuration
datasets = ["bone_tooth_mask"]
split_filename = "bone_tooth_mask"
num_folds = 4
batch_size = 16

# Training Settings
num_epochs = 50

# Model Architecture
[model]
name = "UNet"

[model.parameters]
in_channels = 1
num_classes = 3

# Optimizer
[optimizer]
name = "Adam"

[optimizer.parameters]
lr = 1e-4

# Loss Function
[loss]
name = "MultipleLoss"
main_loss = "Total Loss"

[loss.parameters]
num_classes = 3

# Metric
[metric]
name = "mIoU"

[metric.parameters]
num_classes = 3
```
| You can modify this file or override parameters inside the scripts if needed.

## 🔀 Generate K-Fold Split

```
python -m scripts.prepare_kfold
```
Generates:
```
splits/<SPLIT_FILENAME>.json
{
    "1": {
        "<DATASET_NAME_1>": [
            "data_1",
            ...
        ],
        "<DATASET_NAME_2>": [
            "data_1",
            ...
        ],
        ...
    },
    "2": {...},
    "3": {...},
    "4": {...}
}
```

## 🏋️ Train Model

**Train a single fold**

```
python -m scripts.train --fold 1
```

**Run full experiment (all folds)**

```
python -m scripts.run_experiment
```

Results saved as:
```
📁 logs/<EXPERIMENT_NAME>
├── 📂 Fold_1
│   ├── 📄 best.pth
│   └── 📄 last.pth
├── 📂 Fold_2
├── 📂 Fold_3
├── 📂 Fold_4
├── 📄 <SPLIT_FILENAME>.json
└── 📄 config.toml
```

## 📊 Validation Model Performance

After training is completed, you can run validation to assess model performance on each fold:
```
python -m scripts.evaluate <EXPERIMENT_NAME>
```

This script automatically loads checkpoints for all folds and reports:
* Validation Loss
* Evaluation Metric (e.g., mIoU)

Example Console Output:
```
┌────────┬────────────┬───────────┬────────────────────┬──────────┐
│        │ Total Loss │ Dice Loss │ Cross Entropy Loss │   mIoU   │
├────────┼────────────┼───────────┼────────────────────┼──────────┤
│ Fold 1 │ 0.068619   │ 0.130197  │ 0.007042           │ 0.949898 │
│ Fold 2 │ 0.075404   │ 0.146048  │ 0.004760           │ 0.948226 │
│ Fold 3 │ 0.070967   │ 0.133843  │ 0.008092           │ 0.936442 │
│ Fold 4 │ 0.074820   │ 0.145908  │ 0.003733           │ 0.950953 │
└────────┴────────────┴───────────┴────────────────────┴──────────┘
```

## 🔍 Compare Predictions with Ground Truth

```
python -m scripts.compare <EXPERIMENT_NAME>
```
Outputs:
```
📁 outputs/<EXPERIMENT_NAME>
├── 📂 Fold_1
│   ├── 📂 <DATASET_NAME_1>
│   │   ├── 📂 data_1
│   │   │   ├── 📂 compare
│   │   │   │   ├── 📄 91.png
│   │   │   │   └── 📄 ...
│   │   │   ├── 📂 predict
│   │   │   │   ├── 📄 91.png
│   │   │   │   └── 📄 ...
│   │   │   ├── 📄 ground_truth.npy
│   │   │   └── 📄 volume.npy
│   │   └── 📂 ...
│   └── 📂 <DATASET_NAME_2>
├── 📂 Fold_2
├── 📂 Fold_3
└── 📂 Fold_4
```

## 👁️ Visualize Predictions

You can visualize the segmentation predictions together with ground truth masks:
```
python -m scripts.tools.visualize <EXPERIMENT_NAME>
```

## 🔐 Remote Server Connection

Download experiment logs from a remote server via SFTP.

Create `.env` from the example and fill in your credentials:
```
cp .env.example .env
```
```
SFTP_HOSTNAME = your.server.address
SFTP_PORT = 22
SFTP_USERNAME = your_username
SFTP_PASSWORD = your_password
```
```
python -m scripts.download <EXPERIMENT_NAME>
```

`.env` should not be committed and must be listed in `.gitignore`.

## 📝 Notes

* Ensure the dataset follows the required structure.
* Modify `configs/config.toml` to customize model, loss, optimizer, and metrics.
* The framework is modular and extensible, allowing new models, loss functions, metrics, and optimizers to be added under `src/models/`, `src/losses.py`, `src/metrics.py`, and `src/optimizers.py`.
