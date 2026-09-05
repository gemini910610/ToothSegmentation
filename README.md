# 3D Periodontal Bone Loss Evaluation on CBCT via Image Segmentation and Keypoint Detection

## 📦 Project Structure

```
📁 constants            # Shared enums: Mode (pipeline-stage/.npy filenames), Label (voxel values), Color
📁 dicom                # DICOM loading, HU windowing/resampling, PNG export, patient import
📁 segmentation         # U-Net tooth/bone segmentation: dataset, model, trainer, evaluator, predict
📁 keypoints            # ResNet-18 CEJ keypoint regression: dataset, model, trainer, evaluator, predict (CEJFinder)
📁 panoramic            # 2D panoramic X-ray baseline: vendored YOLOv7 + heatmap keypoint models, predict.py
📁 post_processing      # Batch 3D post-processing stages (connected component → ... → bone loss)
📁 visualize
├── 📂 ui               # Reusable Qt widgets (2D/3D volume viewers, selectors, etc.)
├── 📂 editor           # Manual-correction GUIs: split.py, merge.py, exclude.py, fdi.py
├── 📂 annotator        # keypoints.py — manual CEJ annotation GUI for training data
└── 📂 viewer           # launcher.py — end-to-end single-patient GUI; bone_loss.py — bone-loss inspector
📁 tui                  # Textual TUI building blocks + the argument-collection form for every script
📁 utils                # Config loading, console tables, Unicode-safe image I/O, Excel writer, SFTP transfer, base trainer
📁 scripts              # CLI entry points, one per pipeline step
📁 configs              # segmentation.toml / keypoints.toml — per-task hyperparameters
```

## 🛠️ Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```
uv sync
```

`torch`/`torchvision` are pulled as CUDA 12.6 wheels via a dedicated `uv` index — see `pyproject.toml`.

The panoramic X-ray baseline model needs its weight files placed under `panoramic/weights/`:
```
📁 panoramic/weights
├── 📄 pano_best32type_nolr.pt
├── 📄 pano_transpose_v2_5p.pkl
└── 📄 pano_transpose_v2_6p.pkl
```

## 🔐 Remote Server Connection (SFTP)

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
`.env` is git-ignored and must not be committed.
```
uv run -m scripts.transfer
```

## ⚡ Running Scripts

Every script in `scripts/`, `post_processing/`, `panoramic/predict.py`, and the `visualize.*`
apps is invoked as a module:
```
uv run -m <module.path> [args...]
```

---

## 🔀 Two Ways to Run the Pipeline

* **Interactive, single patient** — `uv run -m visualize.viewer.launcher`, caching each stage's output under `working/<patient_id>/`. See [§ Interactive Viewer](#-interactive-end-to-end-viewer).
* **Batch, many patients** — the chain below, driven by `scripts.*` and `post_processing.*`, writing per-patient `.npy` volumes under `outputs/<segmentation_exp>/<dataset>/<patient>/`.

---

## 📁 Dataset Preparation

### Validate Raw Data Folder Structure

```
uv run -m scripts.check_data <data_dir>
```
Validates raw folder naming against embedded DICOM/X-ray metadata and records each `<group>`'s patients into `datasets/data.xlsx`.

### DICOM → PNG

```
uv run -m scripts.dcm2png <RAW_DATA_DIR> [--min -1000] [--max 4500] [--spacing 0.25]
```
* `<RAW_DATA_DIR>`: raw data root shaped like `<group>/<patient>/<date> CBCT` (see [§ Validate Raw Data Folder Structure](#validate-raw-data-folder-structure)).
* `--min`/`--max`: HU intensity window used before normalization.
* `--spacing`: target voxel spacing (mm) used to resample the volume.

```
📁 datasets/<GROUP_NAME>
└── 📂 <PatientID>_<CBCTDate>
    └── 📂 image
        ├── 📄 1.png
        └── 📄 ...
```

### Segmentation dataset layout

```
📁 datasets/<DATASET_NAME>
├── 📂 image
│   ├── 📂 <patient>_<date>/91.png ...
│   └── 📂 ...
└── 📂 mask
    ├── 📂 <patient>_<date>/91.png ...
    └── 📂 ...
```
* Image and mask folders/filenames must match exactly.
* Mask pixels use labels `{0=background, 1=tooth, 2=bone}`.
* A dataset with a `mask/` subfolder is used for training; one without is inference-only.

### Keypoints dataset layout

```
📁 datasets/slices/<DATASET_NAME>/<patient>_<date>/<LABEL>/{image,segmentation}_<DEGREE>.png
📄 datasets/slices/keypoints.json      # ground-truth CEJ points, produced by the annotator GUI
```
Produced by `post_processing.tooth_slice` and `visualize.annotator.keypoints` — see [§ Post-processing Pipeline](#-post-processing-pipeline).

### Panoramic X-ray layout (baseline comparison)

```
📁 datasets/<DATASET_NAME>/<patient>_<date>/panoramic/*.jpg
```
Collected from raw data via `uv run -m scripts.collect_panoramic <raw_root>` — placed alongside `image`/`mask` under the same `<patient>_<date>` folder as the segmentation dataset.

### Generate K-Fold Split

```
uv run -m scripts.prepare_kfold <segmentation|keypoints>
```
Writes `splits/<split_filename>.json`:
```json
{
    "1": { "<DATASET_NAME_1>": ["<patient>_<date>", "..."], "<DATASET_NAME_2>": ["..."] },
    "2": {}, "3": {}, "4": {}
}
```

## ⚙️ Configuration

* `configs/segmentation.toml`: U-Net, DiceCELoss, mIoU
* `configs/keypoints.toml`: ResNet-18, SmoothL1Loss, PixelError

```toml
experiment = "Segmentation"
num_workers = 4

split_filename = "segmentation"
num_folds = 4

num_epochs = 50
batch_size = 16

[model]
name = "UNet"
[model.parameters]
in_channels = 1
num_classes = 3

[optimizer]
name = "Adam"
[optimizer.parameters]
lr = 1e-4

[loss]
name = "DiceCELoss"
[loss.parameters]
num_classes = 3

[metric]
name = "mIoU"
[metric.parameters]
num_classes = 3
```

## 🏋️ Train Models

`<task>` = `segmentation | keypoints`.

**Find the largest batch size that fits in GPU memory**
```
uv run -m scripts.optimize_batch_size <task> [--is-train] [--maximize]
```
* `--is-train`: profile forward + backward; otherwise profile forward-only.
* `--maximize`: keep doubling until CUDA OOM; otherwise just confirm the configured `batch_size` fits, halving until it does.

**Train a single fold**
```
uv run -m scripts.train <task> <fold>
```
```
📁 logs/<experiment>/Fold_<fold>
├── 📄 best.pth
└── 📄 events.out.tfevents...
```

**Run the full experiment (all folds)**
```
uv run -m scripts.run_experiment <task>
```
```
📁 logs/<experiment>
├── 📂 Fold_1
│   ├── 📄 best.pth
│   └── 📄 events.out.tfevents...
├── 📂 Fold_2 / Fold_3 / Fold_4
├── 📄 <split_filename>.json
└── 📄 <task>.toml
```

## 📊 Evaluate a Trained Experiment

```
uv run -m scripts.evaluate <task> <experiment>
```
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

## 🔍 Batch Segmentation Inference

```
uv run -m scripts.segment <segmentation_exp> <dataset...> [--segmentation-fold FOLD]
```
* `--segmentation-fold`: checkpoint used for patients outside the split (e.g. unlabeled datasets).

```
📁 outputs/<segmentation_exp>/<dataset>/<patient>
├── 📄 image.npy
└── 📄 predict.npy
```

## 🧩 Post-processing Pipeline

| Order | Command | Manual editor | Output |
|---|---|---|---|
| 1 | `uv run -m post_processing.connected_component <segmentation_exp> <dataset...> <target> <threshold>`<br>`target`: `tooth` or `bone` | — | `tooth_cc.npy` / `bone_cc.npy` |
| 2 | `uv run -m post_processing.watershed <segmentation_exp> <dataset...>` | — | `watershed.npy` |
| 3 | `uv run -m post_processing.refine_split <segmentation_exp> <dataset...>` | `uv run -m visualize.editor.split <segmentation_exp> <dataset...>` → `split/<segmentation_exp>.json` | `refine_split.npy` |
| 4 | `uv run -m post_processing.refine_merge <segmentation_exp> <dataset...>` | `uv run -m visualize.editor.merge <segmentation_exp> <dataset...>` → `merge/<segmentation_exp>.json` | `refine_merge.npy` |
| 5 | `uv run -m post_processing.exclude_truncated <segmentation_exp> <dataset...>` | — | `untruncated.npy` |
| 6 | `uv run -m post_processing.exclude_tooth <segmentation_exp> <dataset...>` | `uv run -m visualize.editor.exclude <segmentation_exp> <dataset...>` → `exclude/<segmentation_exp>.json` | `cleaned.npy` |
| 7 | `uv run -m post_processing.fill_holes <segmentation_exp> <dataset...>` | — | `tooth_filled.npy` / `bone_filled.npy` |
| 8 | `uv run -m post_processing.relabel <segmentation_exp> <dataset...>` | — | `relabeled.npy` |
| 9 | `uv run -m post_processing.relabel_fdi <segmentation_exp> <dataset...>` | `uv run -m visualize.editor.fdi <segmentation_exp> <dataset...>` → `fdi/<segmentation_exp>.json` | `fdi.npy` |

* `--threshold` (`connected_component`): voxel-count cutoff for the connected-component filter.

```
📁 outputs/<segmentation_exp>/<dataset>/<patient>
├── 📄 image.npy             # from scripts.segment
├── 📄 predict.npy           # from scripts.segment
├── 📄 tooth_cc.npy          # 1. connected_component
├── 📄 bone_cc.npy           # 1. connected_component
├── 📄 watershed.npy         # 2. watershed
├── 📄 refine_split.npy      # 3. refine_split
├── 📄 refine_merge.npy      # 4. refine_merge
├── 📄 untruncated.npy       # 5. exclude_truncated
├── 📄 cleaned.npy           # 6. exclude_tooth
├── 📄 tooth_filled.npy      # 7. fill_holes
├── 📄 bone_filled.npy       # 7. fill_holes
├── 📄 relabeled.npy         # 8. relabel
└── 📄 fdi.npy               # 9. relabel_fdi
```
```
📄 split/<segmentation_exp>.json     # visualize.editor.split
📄 merge/<segmentation_exp>.json     # visualize.editor.merge
📄 exclude/<segmentation_exp>.json   # visualize.editor.exclude
📄 fdi/<segmentation_exp>.json       # visualize.editor.fdi
```

**Per-tooth oblique slices**
```
uv run -m post_processing.tooth_slice <segmentation_exp> <dataset...>
```
```
📁 outputs/Slice/<segmentation_exp>/<dataset>/<patient>/<label>
├── 📄 image_0.png
├── 📄 segmentation_0.png
├── 📄 image_45.png
├── 📄 segmentation_45.png
└── 📄 ...
```

**Manually annotate CEJ points for keypoints training**
```
uv run -m visualize.annotator.keypoints <segmentation_exp> <dataset...>
```
Writes/updates `datasets/slices/keypoints.json` (same layout as in [§ Dataset Preparation](#-dataset-preparation)).

**Crop panoramic X-ray per tooth**
```
uv run -m post_processing.panoramic <segmentation_exp> <dataset...>
```
```
📁 pano_crop/<segmentation_exp>/<dataset>/<patient>
├── 📄 11.png
├── 📄 12.png
└── 📄 ...
```

## 🦷 Panoramic Baseline Prediction

```
uv run -m panoramic.predict <dataset...>
```
```
📁 outputs/Panoramic/<DATASET_NAME>
└── 📂 <patient>
    └── 📄 <image_stem>.txt
```

## 📈 Bone Loss Computation & Statistical Comparison

```
uv run -m scripts.compare_bone_loss <segmentation_exp> <keypoints_exp> <dataset...> [--keypoints-fold FOLD]
```
* `--keypoints-fold`: fold of the keypoints model to load.
* Runs over patients with `fdi.npy` (post-processing completed).

```
📁 bone_loss
└── 📄 <segmentation_exp>.xlsx
```

```
uv run -m scripts.correlation <segmentation_exp> [--exclude slice|tooth] [--single-only]
```
* `<segmentation_exp>`: the segmentation experiment name the target spreadsheet is named after (`bone_loss/<segmentation_exp>.xlsx`).
* `--exclude slice|tooth`: drop specific angles (`slice`) or whole teeth (`tooth`) listed in `exclude/<segmentation_exp>.json` before recomputing the mean/max bone loss.
* `--single-only`: restrict the comparison to single-rooted teeth (excludes molars).

## 🖥️ Interactive End-to-End Viewer

```
uv run -m visualize.viewer.launcher
```
The launch form collects **"segmentation experiment"**/**"segmentation fold"** and **"keypoints experiment"**/**"keypoints fold"** before opening the per-patient, tab-by-tab GUI (backed by `working/<patient_id>/`):

1. **Import** — bring in a new patient's DICOM series and panoramic photo.
2. **Predict** — run the segmentation model over the CBCT slices.
3. **Bone** / **Tooth** — connected-component filtering (+ automatic watershed splitting for teeth).
4. **Split** / **Merge** — manually correct any watershed under/over-splitting.
5. **Exclude** — mark truncated/unerupted/residual-root/fake teeth, then fill holes and relabel.
6. **Panoramic** — run the 2D panoramic baseline on the patient's panoramic photo.
7. **FDI** — manually assign real FDI tooth numbers, cross-referencing the panoramic prediction.
8. **Bone Loss** — per-tooth CEJ detection and contour/area bone-loss inspection.

```
📁 working/<patient_id>
├── 📂 image/*.png              # 1. Import — imported CBCT slices
├── 📂 panoramic/*.jpg          # 1. Import — imported panoramic photo
├── 📄 image.npy                # 2. Predict
├── 📄 predict.npy              # 2. Predict
├── 📄 tooth_cc.npy             # 3. Tooth
├── 📄 bone_cc.npy              # 3. Bone
├── 📄 watershed.npy            # 3. Tooth (automatic)
├── 📄 refine_split.npy         # 4. Split
├── 📄 untruncated.npy          # 5. Merge (+ truncation check)
├── 📄 tooth_filled.npy         # 5. Exclude
├── 📄 bone_filled.npy          # 5. Exclude
├── 📄 relabeled.npy            # 5. Exclude
├── 📄 panoramic_predict.txt    # 6. Panoramic
└── 📄 fdi.npy                  # 7. FDI
```

Standalone read-only inspector — collects **"segmentation experiment"**, **"keypoints experiment"**, and **"datasets"**:
```
uv run -m visualize.viewer.bone_loss
```
