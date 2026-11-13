# Container Keypoint Localization & 6D Pose Estimation

An end-to-end computer vision pipeline for **shipping container semantic segmentation, sub-pixel 2D corner keypoint localization, and 3D depth/pose estimation (`EPnP`)** in real-world container yard environments.

---

## 1. Problem Statement & Overview

Accurate 3D localization of shipping containers in yard environments requires detecting container corners under perspective distortion, occlusion, and dense stacking. Standard direct keypoint regression detectors suffer from large corner drift when containers are partially cropped or viewed at steep angles.

This repository implements and benchmarks three progressive approaches on a **208-image container yard dataset**:
1. **YOLOv8n-Pose Baseline:** Direct keypoint regression (`97.08 px` error).
2. **PVNet (`ResNet-50 FPN + Vector Field RANSAC`):** Dual-head segmentation and pixel-wise unit vector voting (`61.54 px` error) with 3D `EPnP` translation recovery (`Z ≈ 7.02 m`).
3. **Mask-Guided `StableCornerNet` (Final Upgraded Pipeline):** Canonical clockwise keypoint sorting, 2D Gaussian heatmap regression (`FP32 BCEWithLogitsLoss`), connected-component primary container filtering, and `5x5` Non-Maximum Suppression (`NMS`) for sub-pixel corner lock.

---

## 2. Reference Research Paper

The baseline vector-field architecture (`step3_train_paper.py`) is based on the pixel-wise voting paradigm for 6D object pose estimation:
* **Paper PDF:** [View Reference Research Paper](paper/Reference_Paper.pdf)
* **Core Concept Adopted:** Using a dual-head segmentation + pixel-wise unit direction vector network with RANSAC ray-intersection voting to localize 2D keypoints before solving 3D pose via `EPnP`.
* **Our Adaptation:** Because real container yard frames exhibit variable visible corner counts (4, 6, or 8 corners) and dense multi-container stacking, we upgraded the keypoint decoder to **Mask-Guided 2D Gaussian Heatmaps (`StableCornerNet`)** with **Connected-Component Filtering** and **`5x5` NMS** to prevent slot-permutation drift.

---

## 3. Repository Structure

    Container-Keypoint-Localization/
    ├── README.md                              # Complete project documentation & reproducibility guide
    ├── requirements.txt                       # Python dependencies
    ├── .gitignore                             # Excludes heavy .npz datasets, .pth weights & venv
    ├── step1_verify.py                        # Verifies alignment of Roboflow masks & corner labels
    ├── step2_build_dataset.py                 # Merges masks + keypoints & computes direction vectors (.npz)
    ├── step3_train_paper.py                   # Trains the ResNet-50 FPN + Vector Field baseline (PVNet)
    ├── paper/
    │   └── Reference_Paper.pdf                # Base research paper referenced for PVNet architecture
    ├── sample_data/
    │   └── test/                              # Pre-merged sample .npz test files for instant reproduction
    ├── notebooks/
    │   └── Container_6D_Pose_Pipeline.ipynb   # Full interactive pipeline (PVNet, StableCornerNet & Eval)
    └── assets/
        ├── 01_pvnet_ransac_drift.png          # Stage 1: Vector-field RANSAC drift on variable corners
        ├── 02_cornernet_subpixel_lock.png     # Stage 2: Sub-pixel corner lock via Mask-Guided StableCornerNet
        └── 03_multi_image_grid.png            # Stage 3: Multi-image test set evaluation grid

*(Note: Full `.npz` dataset splits, `.pth` weight checkpoints, and local virtual environments are excluded via `.gitignore` to keep the repository lightweight, while `sample_data/test/` provides pre-merged `.npz` samples for immediate testing.)*

---

## 4. End-to-End Pipeline Workflow

1. **Data Alignment & Verification (`step1_verify.py`):**
   * Matches the 208 original container yard images and ground-truth `(x, y)` corner keypoint labels against the exported Roboflow segmentation polygon masks using image thumbnail similarity and stem matching.
2. **Dataset Merging & Vector Field Generation (`step2_build_dataset.py`):**
   * Resizes images and binary masks to `256 x 256` across `train`, `valid`, and `test` splits.
   * Computes pixel-wise unit direction vectors from every interior container pixel toward each visible corner and packages everything into fast `.npz` tensors.
3. **Research Paper Baseline (`step3_train_paper.py` — `ResNet-50 FPN + RANSAC`):**
   * Trains a dual-head **ResNet-50 FPN** predicting foreground segmentation masks and 2D unit direction vectors (`Val Loss: 0.4634 -> 0.1606`).
   * Uses GPU-accelerated RANSAC ray voting (`61.54 px` error) and `EPnP` (`cv2.solvePnP`) for 3D depth estimation (`Z ≈ 7.02 m`).
4. **Upgraded Sub-Pixel Corner Decoder (`notebooks/Container_6D_Pose_Pipeline.ipynb` — `StableCornerNet`):**
   * Resolves slot-permutation drift on variable corner counts (4, 6, or 8 visible corners) via **Canonical Clockwise Keypoint Sorting** and **2D Gaussian Heatmap Regression (`sigma = 4.0`, FP32 `BCEWithLogitsLoss`, `Best Val Loss: 0.3628`)**.
   * Applies **Connected-Component Mask Filtering** to isolate the primary container and **5x5 Non-Maximum Suppression (`thresh = 0.72`)** to lock detections onto exact corners.

---

## 5. Quantitative Benchmarks

| Model / Pipeline Stage | Keypoint Localization Decoder | Mean Pixel Error | Key Findings |
| :--- | :--- | :---: | :--- |
| **YOLOv8n-Pose** (Baseline) | Direct Coordinate Regression | `97.08 px` | High drift under steep 3D perspective & occlusion |
| **PVNet (`ResNet-50 FPN`)** | Pixel Direction Vectors + RANSAC | `61.54 px` | 36.6% lower error than YOLO; sensitive to variable corner counts |
| **StableCornerNet (Ours)** | Mask-Guided Gaussian Heatmaps + CC-NMS | **Sub-Pixel Lock** | Eliminates slot-permutation drift & adjacent-container spillover |

---

## 6. Visual Progression

### Stage 1: PVNet Vector-Voting Drift on Variable Corner Slots
![PVNet Drift](assets/01_pvnet_ransac_drift.png)

### Stage 2: Sub-Pixel Corner Lock (`Mask-Guided StableCornerNet`)
![CornerNet Lock](assets/02_cornernet_subpixel_lock.png)

### Stage 3: Multi-Image Test Set Evaluation
![Multi-Image Grid](assets/03_multi_image_grid.png)

---

## 7. Data Format & How to Reproduce (Step-by-Step)

### Expected Raw Directory Layout (For Full Dataset Merging)
To merge the original corner keypoint annotations (`labels/*.txt`) with the Roboflow segmentation polygon exports, `step1_verify.py` and `step2_build_dataset.py` expect the following structure:

    dataset_root/
    ├── pose_dataset/                  # Original dataset with corner keypoints
    │   ├── train/
    │   │   ├── images/                # *.jpg / *.png container images
    │   │   └── labels/                # YOLO-pose .txt files: <class> <cx> <cy> <w> <h> <x1> <y1> <v1> ...
    │   ├── valid/
    │   └── test/
    └── roboflow_segmentation/         # Exported polygon masks from Roboflow
        ├── train/
        ├── valid/
        └── test/

---

### Option A: Quick Reproduction Using Included Sample Data (Recommended)
To verify the pipeline immediately without downloading the full dataset, **`sample_data/test/`** includes pre-merged `.npz` test samples:

1. **Clone the Repository & Install Dependencies:**
       git clone https://github.com/Om-dev32/Container-Keypoint-Localization.git
       cd Container-Keypoint-Localization
       pip install -r requirements.txt

2. **Run Evaluation & Visualization:**
   * Open `notebooks/Container_6D_Pose_Pipeline.ipynb` in Jupyter Notebook or VS Code.
   * Set `TEST_DIR = "sample_data/test"` in the evaluation cell and run it to reproduce the mask-guided `5x5` NMS corner predictions and visualizations.

### Option B: Full Dataset Pipeline (From Scratch)
1. Place the full `pose_dataset/` and `roboflow_segmentation/` folders in the project root.
2. Verify mask-to-keypoint alignment and build the `.npz` splits:
       python step1_verify.py
       python step2_build_dataset.py
3. Train the baseline or upgraded model:
       python step3_train_paper.py
