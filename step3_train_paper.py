import os
import glob
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

DATA_DIR = r"C:\Users\om\OneDrive\Desktop\exotrac_paper\processed_data"
SAVE_DIR = r"C:\Users\om\OneDrive\Desktop\exotrac_paper\checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

IMG_SIZE = 256   # 512 -> 256: 4x kam VRAM, 15x fast training!
MAX_KPTS = 8
BATCH_SIZE = 8
EPOCHS = 60
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------
# 1. Fast In-Memory Dataset Loader (Resized to 256x256)
# ---------------------------------------------------------
class ContainerPoseSegDataset(Dataset):
    def __init__(self, split_dir, augment=False):
        files = sorted(glob.glob(os.path.join(split_dir, "*.npz")))
        self.augment = augment
        self.samples = []
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        for f in files:
            d = np.load(f)
            orig_h, orig_w = d["sem_mask"].shape
            scale = IMG_SIZE / float(orig_h)

            img = torch.from_numpy(d["image"].astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
            img = F.interpolate(img, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False).squeeze(0).permute(1, 2, 0).numpy()
            img = img[:, :, ::-1].copy()  # BGR -> RGB

            sem = torch.from_numpy(d["sem_mask"].astype(np.float32)).unsqueeze(0).unsqueeze(0)
            sem = F.interpolate(sem, size=(IMG_SIZE, IMG_SIZE), mode="nearest").squeeze().long().numpy()

            vf = torch.from_numpy(d["vec_field"].astype(np.float32)).unsqueeze(0)
            vf = F.interpolate(vf, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False).squeeze(0).numpy()

            vm = torch.from_numpy(d["vec_mask"].astype(np.float32)).unsqueeze(0)
            vm = F.interpolate(vm, size=(IMG_SIZE, IMG_SIZE), mode="nearest").squeeze(0).numpy()

            kpts = d["kpts"].astype(np.float32).copy()
            valid = kpts[:, 0] >= 0
            kpts[valid] *= scale

            self.samples.append((img, sem, vf, vm, kpts, mean, std))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, sem_mask, vec_field, vec_mask, kpts, mean, std = self.samples[idx]
        if self.augment and np.random.rand() < 0.5:
            img = np.clip(img * np.random.uniform(0.8, 1.2), 0.0, 1.0)
        img = np.transpose((img - mean) / std, (2, 0, 1))
        return (
            torch.from_numpy(img),
            torch.from_numpy(sem_mask),
            torch.from_numpy(vec_field),
            torch.from_numpy(vec_mask),
            torch.from_numpy(kpts),
        )

# ---------------------------------------------------------
# 2. Research Paper Network (ResNet-50 + Feature Pyramid)
# ---------------------------------------------------------
class ResNet50FPNPoseNet(nn.Module):
    def __init__(self, num_classes=2, num_kpts=8):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.lat1 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.lat2 = nn.Sequential(nn.Conv2d(512, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.lat3 = nn.Sequential(nn.Conv2d(1024, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.lat4 = nn.Sequential(nn.Conv2d(2048, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.seg_head = nn.Conv2d(256, num_classes, kernel_size=1)
        self.vec_head = nn.Conv2d(256, num_kpts * 2, kernel_size=1)

    def forward(self, x):
        c1 = self.layer1(self.stem(x))
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)

        p1 = self.lat1(c1)
        p2 = F.interpolate(self.lat2(c2), size=p1.shape[2:], mode="bilinear", align_corners=False)
        p3 = F.interpolate(self.lat3(c3), size=p1.shape[2:], mode="bilinear", align_corners=False)
        p4 = F.interpolate(self.lat4(c4), size=p1.shape[2:], mode="bilinear", align_corners=False)

        feat = torch.cat([p1, p2, p3, p4], dim=1)
        feat = self.up2(self.up1(feat))
        return self.seg_head(feat), self.vec_head(feat)

# ---------------------------------------------------------
# 3. Paper Loss Function: L = l1*L_seg + L_vf + l2*L_pv
# ---------------------------------------------------------
class PaperVotingLoss(nn.Module):
    def __init__(self, img_size=IMG_SIZE, num_kpts=8):
        super().__init__()
        self.num_kpts = num_kpts
        ys, xs = torch.meshgrid(torch.arange(img_size), torch.arange(img_size), indexing="ij")
        self.register_buffer("xs", xs.float())
        self.register_buffer("ys", ys.float())

    def forward(self, seg_logits, vec_pred, sem_mask, vec_gt, vec_mask, kpts, lam1=1.0, lam2=1e-3):
        l_seg = F.cross_entropy(seg_logits, sem_mask)

        B, _, H, W = vec_pred.shape
        vp = vec_pred.view(B, self.num_kpts, 2, H, W)
        vp_unit = vp / (torch.norm(vp, dim=2, keepdim=True) + 1e-6)

        vg = vec_gt.view(B, self.num_kpts, 2, H, W)
        vm = vec_mask.unsqueeze(2)

        valid_pixels = vm.sum() * 2.0 + 1e-6
        l_vf = F.smooth_l1_loss(vp_unit * vm, vg * vm, reduction="sum") / valid_pixels

        vx = vp_unit[:, :, 0]
        vy = vp_unit[:, :, 1]
        px = self.xs.view(1, 1, H, W)
        py = self.ys.view(1, 1, H, W)
        kx = kpts[:, :, 0].view(B, self.num_kpts, 1, 1)
        ky = kpts[:, :, 1].view(B, self.num_kpts, 1, 1)

        valid_kpt = (kpts[:, :, 0] >= 0).float().view(B, self.num_kpts, 1, 1) * vec_mask
        lam_prime = (vx * (kx - px) + vy * (ky - py)) / (vx**2 + vy**2 + 1e-6)
        dist_perp = torch.sqrt((kx - px - vx * lam_prime) ** 2 + (ky - py - vy * lam_prime) ** 2 + 1e-6)
        dist_norm = dist_perp / float(H)
        l_pv = F.smooth_l1_loss(dist_norm * valid_kpt, torch.zeros_like(dist_norm), reduction="sum") / (valid_kpt.sum() + 1e-6)

        total_loss = lam1 * l_seg + l_vf + lam2 * l_pv
        return total_loss, l_seg.item(), l_vf.item(), l_pv.item()

# ---------------------------------------------------------
# 4. Training Loop
# ---------------------------------------------------------
def train():
    print(f"Training on device: {DEVICE} | Resolution: {IMG_SIZE}x{IMG_SIZE}", flush=True)
    t0 = time.time()
    train_ds = ContainerPoseSegDataset(os.path.join(DATA_DIR, "train"), augment=True)
    val_ds = ContainerPoseSegDataset(os.path.join(DATA_DIR, "valid"), augment=False)
    print(f"Loaded {len(train_ds)} train & {len(val_ds)} val images in {time.time()-t0:.1f}s!", flush=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = ResNet50FPNPoseNet(num_classes=2, num_kpts=MAX_KPTS).to(DEVICE)
    criterion = PaperVotingLoss(img_size=IMG_SIZE, num_kpts=MAX_KPTS).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.75)
    scaler = torch.amp.GradScaler("cuda")

    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        ep_start = time.time()
        model.train()
        lam2 = 1e-3 + (1e-2 - 1e-3) * (epoch / EPOCHS)
        tr_loss, tr_seg, tr_vf, tr_pv = 0.0, 0.0, 0.0, 0.0

        for imgs, masks, vecs, vmasks, kpts in train_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            vecs, vmasks, kpts = vecs.to(DEVICE), vmasks.to(DEVICE), kpts.to(DEVICE)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                seg_out, vec_out = model(imgs)
                loss, ls, lv, lp = criterion(seg_out, vec_out, masks, vecs, vmasks, kpts, lam1=1.0, lam2=lam2)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            tr_loss += loss.item()
            tr_seg += ls
            tr_vf += lv
            tr_pv += lp

        scheduler.step()
        n_tr = len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, masks, vecs, vmasks, kpts in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                vecs, vmasks, kpts = vecs.to(DEVICE), vmasks.to(DEVICE), kpts.to(DEVICE)
                with torch.amp.autocast("cuda"):
                    seg_out, vec_out = model(imgs)
                    loss, _, _, _ = criterion(seg_out, vec_out, masks, vecs, vmasks, kpts, lam1=1.0, lam2=lam2)
                val_loss += loss.item()
        val_loss /= max(1, len(val_loader))

        print(f"Epoch [{epoch:02d}/{EPOCHS}] ({time.time()-ep_start:.1f}s) | Train: {tr_loss/n_tr:.4f} (seg:{tr_seg/n_tr:.3f}, vf:{tr_vf/n_tr:.3f}, pv:{tr_pv/n_tr:.3f}) | Val: {val_loss:.4f}", flush=True)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_paper_model.pth"))

    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "last_paper_model.pth"))
    print(f"Training Finished! Best Val Loss: {best_val_loss:.4f}", flush=True)

if __name__ == "__main__":
    train()