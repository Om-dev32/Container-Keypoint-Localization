import os
import glob
import cv2
import numpy as np

POSE_DIR = r"C:\Users\om\OneDrive\Desktop\Container_classification\custom_dataset"
SEG_DIR = r"C:\Users\om\OneDrive\Desktop\paper seg"
OUT_DIR = r"C:\Users\om\OneDrive\Desktop\exotrac_paper"
os.makedirs(OUT_DIR, exist_ok=True)

def get_base_name(filename):
    # Roboflow ke '_jpg.rf.xxxx' ya '_png.rf.xxxx' ko hata kar original naam nikalta hai
    name = os.path.splitext(filename)[0]
    if ".rf." in name:
        name = name.split(".rf.")[0]
        if name.endswith("_jpg") or name.endswith("_png") or name.endswith("_jpeg"):
            name = name.rsplit("_", 1)[0]
    return name

# 1. Saare Pose labels aur Seg labels ko index karo (across train/valid/test)
pose_labels = {}
for path in glob.glob(os.path.join(POSE_DIR, "*", "labels", "*.txt")):
    pose_labels[get_base_name(os.path.basename(path))] = path

seg_items = {}
for split in ["train", "valid", "test"]:
    for img_path in glob.glob(os.path.join(SEG_DIR, split, "images", "*.*")):
        base = get_base_name(os.path.basename(img_path))
        lbl_name = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
        lbl_path = os.path.join(SEG_DIR, split, "labels", lbl_name)
        if os.path.exists(lbl_path):
            seg_items[base] = (split, img_path, lbl_path)

matched = sorted(set(pose_labels.keys()) & set(seg_items.keys()))
print(f"Total Pose Labels: {len(pose_labels)}")
print(f"Total Seg Images : {len(seg_items)}")
print(f"Matched Both     : {len(matched)}")

# 2. Ek matched image par Mask + Keypoints draw karke verify karo
if matched:
    sample_base = matched[0]
    split, img_path, seg_lbl_path = seg_items[sample_base]
    pose_lbl_path = pose_labels[sample_base]

    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    overlay = img.copy()

    # Draw Segmentation Polygons (Green)
    with open(seg_lbl_path, "r") as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            if len(parts) > 5:
                pts = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
                pts[:, 0] *= w
                pts[:, 1] *= h
                cv2.fillPoly(overlay, [pts.astype(np.int32)], (0, 255, 0))

    img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)

    # Draw Pose Keypoints (Red circles)
    with open(pose_lbl_path, "r") as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            # parts: cls, cx, cy, bw, bh, kpt1_x, kpt1_y, v1, ...
            kpts = parts[5:]
            for i in range(0, len(kpts), 3):
                kx, ky, kv = kpts[i] * w, kpts[i+1] * h, kpts[i+2]
                if kv > 0:
                    cv2.circle(img, (int(kx), int(ky)), 6, (0, 0, 255), -1)

    save_path = os.path.join(OUT_DIR, "verify_overlay.jpg")
    cv2.imwrite(save_path, img)
    print(f"Sample saved at: {save_path} (Image: {sample_base})")