import os
import glob
import cv2
import numpy as np

POSE_DIR = r"C:\Users\om\OneDrive\Desktop\Container_classification\custom_dataset"
SEG_DIR = r"C:\Users\om\OneDrive\Desktop\paper seg"
DATA_OUT = r"C:\Users\om\OneDrive\Desktop\exotrac_paper\processed_data"
os.makedirs(DATA_OUT, exist_ok=True)

IMG_SIZE = 512
MAX_KPTS = 8

def get_prefix(stem):
    # '4_jpg.rf.xxxx' -> '4_jpg'
    return stem.split(".rf.")[0] if ".rf." in stem else stem

def get_thumb(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    small = cv2.resize(img, (32, 32)).astype(np.float32)
    small = (small - small.mean()) / (small.std() + 1e-6)
    return small.flatten()

# 1. Purane Pose dataset ke saare items load karo
pose_items = []
for split in ["train", "valid", "test"]:
    for img_p in glob.glob(os.path.join(POSE_DIR, split, "images", "*.*")):
        stem = os.path.splitext(os.path.basename(img_p))[0]
        lbl_p = os.path.join(POSE_DIR, split, "labels", stem + ".txt")
        if os.path.exists(lbl_p):
            pose_items.append({
                "stem": stem,
                "prefix": get_prefix(stem),
                "lbl": lbl_p,
                "thumb": get_thumb(img_p)
            })

# 2. Match & Build Direction Vectors
stats = {"train": 0, "valid": 0, "test": 0, "unmatched": 0}
ys, xs = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE].astype(np.float32)

for split in ["train", "valid", "test"]:
    split_dir = os.path.join(DATA_OUT, split)
    os.makedirs(split_dir, exist_ok=True)

    for img_p in glob.glob(os.path.join(SEG_DIR, split, "images", "*.*")):
        stem = os.path.splitext(os.path.basename(img_p))[0]
        seg_lbl_p = os.path.join(SEG_DIR, split, "labels", stem + ".txt")
        if not os.path.exists(seg_lbl_p):
            continue

        prefix = get_prefix(stem)
        candidates = [p for p in pose_items if p["prefix"] == prefix]
        if not candidates:
            candidates = pose_items  # Fallback: search all

        seg_thumb = get_thumb(img_p)
        best_match = None
        best_score = -1.0

        for cand in candidates:
            if cand["stem"] == stem:
                best_match = cand
                best_score = 1.0
                break
            if seg_thumb is not None and cand["thumb"] is not None:
                score = float(np.dot(seg_thumb, cand["thumb"]) / 1024.0)
                if score > best_score:
                    best_score = score
                    best_match = cand

        if best_match is None or best_score < 0.75:
            stats["unmatched"] += 1
            continue

        pose_lbl_p = best_match["lbl"]
        img = cv2.imread(img_p)
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        all_kpts = []
        with open(pose_lbl_p, "r") as f:
            for line in f:
                parts = list(map(float, line.strip().split()))
                kpts = parts[5:]
                for i in range(0, len(kpts), 3):
                    kx, ky, kv = kpts[i] * IMG_SIZE, kpts[i+1] * IMG_SIZE, kpts[i+2]
                    if kv > 0:
                        all_kpts.append([kx, ky])
        all_kpts = np.array(all_kpts, dtype=np.float32) if len(all_kpts) > 0 else np.zeros((0, 2), dtype=np.float32)

        sem_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
        vec_field = np.zeros((MAX_KPTS * 2, IMG_SIZE, IMG_SIZE), dtype=np.float32)
        vec_mask = np.zeros((MAX_KPTS, IMG_SIZE, IMG_SIZE), dtype=np.float32)
        saved_kpts = np.full((MAX_KPTS, 2), -1.0, dtype=np.float32)

        with open(seg_lbl_p, "r") as f:
            lines = f.readlines()

        inst_kpts_list = []
        for line in lines:
            parts = list(map(float, line.strip().split()))
            if len(parts) <= 5:
                continue
            pts = np.array(parts[1:], dtype=np.float32).reshape(-1, 2) * IMG_SIZE
            inst_mask = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)
            cv2.fillPoly(inst_mask, [pts.astype(np.int32)], 1)
            sem_mask[inst_mask == 1] = 1

            if len(all_kpts) > 0:
                dilated = cv2.dilate(inst_mask, np.ones((15, 15), np.uint8))
                matched_pts = []
                for kp in all_kpts:
                    ix = int(np.clip(round(kp[0]), 0, IMG_SIZE - 1))
                    iy = int(np.clip(round(kp[1]), 0, IMG_SIZE - 1))
                    if dilated[iy, ix] == 1:
                        matched_pts.append(kp)

                if len(matched_pts) > 0:
                    matched_pts = sorted(matched_pts, key=lambda p: (round(p[1] / 25), p[0]))[:MAX_KPTS]
                    inst_kpts_list.append(matched_pts)

                    fg = inst_mask == 1
                    for k_idx, (kx, ky) in enumerate(matched_pts):
                        dx = kx - xs[fg]
                        dy = ky - ys[fg]
                        dist = np.sqrt(dx**2 + dy**2) + 1e-6
                        vec_field[k_idx * 2][fg] = dx / dist
                        vec_field[k_idx * 2 + 1][fg] = dy / dist
                        vec_mask[k_idx][fg] = 1.0

        if len(inst_kpts_list) > 0:
            for k_idx, (kx, ky) in enumerate(inst_kpts_list[0]):
                saved_kpts[k_idx] = [kx, ky]

        out_file = os.path.join(split_dir, f"{stats[split]:04d}_{prefix[:15]}.npz")
        np.savez_compressed(
            out_file,
            image=img_resized,
            sem_mask=sem_mask,
            vec_field=vec_field,
            vec_mask=vec_mask,
            kpts=saved_kpts
        )
        stats[split] += 1

print("Dataset Build Complete!")
print(stats)