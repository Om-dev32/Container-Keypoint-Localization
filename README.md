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

## 2. Repository Structure

    Container-Keypoint-Localization/
    ├── README.md                              # Complete project documentation & setup guide
    ├── requirements.txt                       # Python dependencies
    ├── .gitignore                             # Excludes heavy .npz datasets & .pth model weights
    ├── notebooks/
    │   └── Container_6D_Pose_Pipeline.ipynb   # End-to-end notebook (Data Merge, PVNet, StableCornerNet, Eval)
    └── assets/
        ├── 01_pvnet_ransac_drift.png          # Stage 1: Vector-field RANSAC drift on variable corners
        ├── 02_cornernet_subpixel_lock.png     # Stage 2: Sub-pixel corner lock via Mask-Guided StableCornerNet
        └── 03_multi_image_grid.png            # Stage 3: Multi-image test set evaluation grid

*(Note: Large `.npz` dataset splits and `.pth` weight checkpoints are excluded via `.gitignore` to keep the repository lightweight. Instructions to regenerate them are provided in Section 6.)*

---

## 3. Dataset Preparation & Data Merging Pipeline

* **Source Data:** 208 container yard images split into `train`, `valid`, and `test` subsets.
* **Annotations Merged:**
  * **Ground-Truth Corner Keypoints:** `(x, y)` coordinates of visible container corners from the original dataset labels.
  * **Segmentation Polygons:** Binary container masks manually annotated and exported from Roboflow.
* **Preprocessing (`.npz` Packaging):**
  * Each image and segmentation mask is resized to `256 x 256`.
  * For **PVNet**, unit direction vectors pointing from every interior foreground pixel toward each corner are precomputed.
  * For **`StableCornerNet`**, visible keypoints are **canonically sorted clockwise** around the container centroid to prevent slot-index permutation across frames, and converted into continuous **2D Gaussian Heatmaps** (`sigma = 4.0`).

---

## 4. Quantitative Benchmarks

| Model / Pipeline Stage | Keypoint Localization Decoder | Mean Pixel Error | Key Findings |
| :--- | :--- | :---: | :--- |
| **YOLOv8n-Pose** (Baseline) | Direct Coordinate Regression | `97.08 px` | High drift under steep 3D perspective & occlusion |
| **PVNet (`ResNet-50 FPN`)** | Pixel Direction Vectors + RANSAC | `61.54 px` | 36.6% lower error than YOLO; sensitive to variable corner counts |
| **StableCornerNet (Ours)** | Mask-Guided Gaussian Heatmaps + CC-NMS | **Sub-Pixel Lock** | Eliminates slot-permutation drift & adjacent-container spillover |

---

## 5. Visual Progression

### Stage 1: PVNet Vector-Voting Drift on Variable Corner Slots
![PVNet Drift](assets/01_pvnet_ransac_drift.png)

### Stage 2: Sub-Pixel Corner Lock (`Mask-Guided StableCornerNet`)
![CornerNet Lock](assets/02_cornernet_subpixel_lock.png)

### Stage 3: Multi-Image Test Set Evaluation
![Multi-Image Grid](assets/03_multi_image_grid.png)

---

## 6. How to Reproduce (Step-by-Step)

### Step 1: Clone the Repository & Install Dependencies
    git clone https://github.com/Om-dev32/Container-Keypoint-Localization.git
    cd Container-Keypoint-Localization
    pip install -r requirements.txt

### Step 2: Prepare Dataset Directory
Place the merged `.npz` dataset splits (`train/`, `valid/`, `test/`) inside a local `processed_data/` folder (each `.npz` file contains `image`, `sem_mask`, and `kpts` arrays):

    processed_data/
    ├── train/   # *.npz files
    ├── valid/   # *.npz files
    └── test/    # *.npz files

### Step 3: Run Training & Evaluation
1. Open `notebooks/Container_6D_Pose_Pipeline.ipynb` in Jupyter Notebook or VS Code.
2. Update `DATA_DIR` and `SAVE_DIR` paths in the configuration cell if running outside the default directory.
3. Run all cells sequentially to:
   * Train `StableCornerNet` for 50 epochs in pure FP32 (`Best Val Loss: 0.3628`, saved to `checkpoints/stable_corner_net.pth`).
   * Run primary-container connected-component filtering + `5x5` NMS (`thresh = 0.72`) across the test set.
   * Generate side-by-side Predicted (Red) vs. Ground Truth (Cyan) visualizations and compute mean localization error and FPS.
