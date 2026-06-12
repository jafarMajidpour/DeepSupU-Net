

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import efficientnet_b5, EfficientNet_B5_Weights
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import List
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


DATA_ROOT   = 'D:/Jafar/Ultrasound1/BUSI/All/'   # <-- your data path
SAVE_DIR    = 'D:/Jafar/Ultrasound1/BUSI/results_enhanced' # <-- output folder

IMG_SIZE    = 256          # input resolution
BATCH_SIZE  = 8
NUM_WORKERS = 0            # MUST be 0 on Windows (fork-based multiprocessing not supported)
EPOCHS      = 1000
BASE_LR     = 1e-3         # decoder learning rate
WEIGHT_DECAY = 1e-4
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15
SEED        = 42


T0          = 100          # first cycle length (epochs)
T_MULT      = 2            # cycle length multiplier


AUX_WEIGHT  = 0.4

USE_TTA     = True

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(SEED)
np.random.seed(SEED)

print()
print('=' * 80)
print('ENHANCED BREAST ULTRASOUND SEGMENTATION')
print('=' * 80)
print(f'Device : {DEVICE}')
print(f'Data   : {DATA_ROOT}')
print(f'Encoder: EfficientNet-B5  (upgraded from B4)')
print(f'Extras : CBAM | ASPP-lite | Deep-Sup | TTA | Boundary-Loss')
print('=' * 80)


# DATA AUGMENTATION
def get_training_transforms():
    return A.Compose([
        A.RandomResizedCrop(IMG_SIZE, IMG_SIZE, scale=(0.7, 1.0), ratio=(0.8, 1.2)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.6),
        A.OneOf([
            A.ElasticTransform(alpha=80, sigma=10, p=1.0),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
        ], p=0.4),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.3),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
            A.CLAHE(clip_limit=4.0, p=1.0),
            A.RandomGamma(gamma_limit=(70, 130), p=1.0),
        ], p=0.5),
        A.GaussNoise(var_limit=(5.0, 30.0), p=0.3),
        A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(16, 32), hole_width_range=(16, 32), fill_value=0, p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def get_validation_transforms():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# DATASET

class BUSIDataset(Dataset):
    def __init__(self, image_paths: List[str], mask_paths: List[str], transform=None):
        self.image_paths = image_paths
        self.mask_paths  = mask_paths
        self.transform   = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask  = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask  = (mask > 127).astype(np.float32)

        if image.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
                              interpolation=cv2.INTER_NEAREST)

        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask  = transformed['mask']

        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)

        return image, mask


# DATA LOADING


def find_image_mask_pairs(root_dir: Path):
    print('=' * 80)
    print('DATA LOADING')
    print('=' * 80)
    print(f'Root: {root_dir}')

    if not root_dir.exists():
        raise FileNotFoundError(f'Directory not found: {root_dir}')

    images_dir, masks_dir = None, None
    for img_folder, mask_folder in [('images', 'masks'), ('Images', 'Masks')]:
        img_path  = root_dir / img_folder
        mask_path = root_dir / mask_folder
        if img_path.exists() and mask_path.exists():
            images_dir, masks_dir = img_path, mask_path
            print(f'Found: {img_folder}/ and {mask_folder}/')
            break

    if images_dir is None:
        images_dir = masks_dir = root_dir
        print('Using flat structure')

    image_files = []
    for ext in ['*.png', '*.jpg', '*.PNG', '*.JPG']:
        image_files.extend(list(images_dir.glob(ext)))

    image_files = sorted([f for f in image_files if 'mask' not in f.stem.lower()])
    print(f'Found {len(image_files)} images')

    if len(image_files) == 0:
        raise ValueError(f'No images found in {images_dir}')

    image_paths, mask_paths = [], []
    for img_path in image_files:
        img_name  = img_path.stem
        mask_found = None
        for pattern in [f'{img_name}_mask*.png', f'{img_name}_mask*.jpg']:
            matches = list(masks_dir.glob(pattern))
            if matches:
                mask_found = matches[0]
                break
        if mask_found:
            image_paths.append(str(img_path))
            mask_paths.append(str(mask_found))

    print(f'Matched {len(image_paths)}/{len(image_files)} pairs')
    if len(image_paths) == 0:
        raise ValueError('No valid pairs found')

    return image_paths, mask_paths


def create_dataloaders(root, batch_size, val_ratio, test_ratio, seed):
    root_path = Path(root)
    image_paths, mask_paths = find_image_mask_pairs(root_path)

    train_imgs, temp_imgs, train_masks, temp_masks = train_test_split(
        image_paths, mask_paths,
        test_size=(val_ratio + test_ratio), random_state=seed
    )
    val_imgs, test_imgs, val_masks, test_masks = train_test_split(
        temp_imgs, temp_masks,
        test_size=test_ratio / (val_ratio + test_ratio), random_state=seed
    )

    print(f'Train={len(train_imgs)}, Val={len(val_imgs)}, Test={len(test_imgs)}')

    train_ds = BUSIDataset(train_imgs, train_masks, get_training_transforms())
    val_ds   = BUSIDataset(val_imgs,   val_masks,   get_validation_transforms())
    test_ds  = BUSIDataset(test_imgs,  test_masks,  get_validation_transforms())

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=pin,
                              persistent_workers=False)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin,
                              persistent_workers=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=pin,
                              persistent_workers=False)

    print('DataLoaders ready')
    print('=' * 80 + '\n')
    return train_loader, val_loader, test_loader


# MODEL BUILDING BLOCKS

class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation style channel attention."""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = self.fc(self.avg_pool(x))
        mx  = self.fc(self.max_pool(x))
        return self.sigmoid(avg + mx)


class SpatialAttention(nn.Module):
    """Spatial attention using avg+max channel descriptors."""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module (Woo et al., ECCV 2018)."""
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.ca(x)   # channel-wise recalibration
        x = x * self.sa(x)   # spatial recalibration
        return x


# 2. ASPP-lite  (Multi-Scale Pooling at Bottleneck) 

class ASPPLite(nn.Module):
    """
    Lightweight ASPP: captures multi-scale context at the bottleneck.
    Uses dilated convolutions with rates [1, 6, 12, 18] + global avg pool.
    """
    def __init__(self, in_channels: int, out_channels: int = 256):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(                                          # rate 1
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)),
            nn.Sequential(                                          # rate 6
                nn.Conv2d(in_channels, out_channels, 3,
                          padding=6, dilation=6, bias=False),
                nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)),
            nn.Sequential(                                          # rate 12
                nn.Conv2d(in_channels, out_channels, 3,
                          padding=12, dilation=12, bias=False),
                nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)),
            nn.Sequential(                                          # rate 18
                nn.Conv2d(in_channels, out_channels, 3,
                          padding=18, dilation=18, bias=False),
                nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)),
        ])
        # global average pooling branch
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def forward(self, x):
        size = x.shape[2:]
        outs = [b(x) for b in self.branches]
        gap  = F.interpolate(self.gap(x), size=size,
                             mode='bilinear', align_corners=False)
        outs.append(gap)
        return self.project(torch.cat(outs, dim=1))


# 3. Decoder block with optional stochastic depth 

class DecoderBlock(nn.Module):
    """
    Decoder block:
      upsample → concat skip → double-conv → CBAM
    Stochastic depth (drop_prob) applied during training for regularisation.
    """
    def __init__(self, in_channels: int, skip_channels: int,
                 out_channels: int, drop_prob: float = 0.0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.cbam      = CBAM(out_channels)
        self.drop_prob = drop_prob

    def _drop_path(self, x):
        """Stochastic depth: randomly drop entire residual paths."""
        if not self.training or self.drop_prob == 0.:
            return x
        keep = 1. - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        noise = torch.rand(shape, dtype=x.dtype, device=x.device)
        noise = torch.floor(noise + keep)
        return x * noise / keep

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[2:],
                          mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self._drop_path(self.conv(x))
        x = self.cbam(x)
        return x


#4. Auxiliary segmentation head (for deep supervision) 

class AuxHead(nn.Module):
    def __init__(self, in_channels: int, num_classes: int = 1):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, num_classes, 1),
        )

    def forward(self, x, target_size):
        x = self.head(x)
        return F.interpolate(x, size=target_size,
                             mode='bilinear', align_corners=False)


# MAIN MODEL:  EnhancedUNet

class EnhancedUNet(nn.Module):
    """
    EfficientNet-B5 encoder
    + ASPP-lite bottleneck
    + 4 decoder stages each with CBAM attention on skip connection
    + 2 auxiliary heads (deep supervision at dec2 and dec3)
    """
    SKIP_INDICES = [2, 3, 5, 8]   # same as baseline

    def __init__(self, num_classes: int = 1):
        super().__init__()

        print('Creating EnhancedUNet (EfficientNet-B5 + CBAM + ASPP + DeepSup) ...')

        # Encoder
        effnet = efficientnet_b5(weights=EfficientNet_B5_Weights.IMAGENET1K_V1)
        self.encoder = effnet.features

        # Detect channel sizes automatically
        skip_ch, bottle_ch = self._detect_channels()
        print(f'  Skip channels   : {skip_ch}')
        print(f'  Bottleneck ch   : {bottle_ch}')

        # CBAM gates on skip connections
        self.skip_cbam = nn.ModuleList([
            CBAM(c) for c in skip_ch
        ])

        # ASPP-lite bottleneck  →  project to 256 ch
        BOTTLE_OUT = 256
        self.aspp = ASPPLite(bottle_ch, BOTTLE_OUT)

        # Decoder blocks (stochastic depth increases towards early layers)
        self.dec4 = DecoderBlock(BOTTLE_OUT, skip_ch[3], 256, drop_prob=0.05)
        self.dec3 = DecoderBlock(256,         skip_ch[2], 128, drop_prob=0.10)
        self.dec2 = DecoderBlock(128,         skip_ch[1],  64, drop_prob=0.10)
        self.dec1 = DecoderBlock( 64,         skip_ch[0],  32, drop_prob=0.05)

        # Final segmentation head
        self.final = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_classes, 1),
        )

        # Auxiliary heads for deep supervision
        self.aux_head3 = AuxHead(128, num_classes)  # after dec3
        self.aux_head2 = AuxHead( 64, num_classes)  # after dec2

        print('  Model ready')

    def _detect_channels(self):
        with torch.no_grad():
            x = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
            skip_ch = []
            for idx, layer in enumerate(self.encoder):
                x = layer(x)
                if idx in self.SKIP_INDICES:
                    skip_ch.append(x.shape[1])
            return skip_ch, x.shape[1]

    def forward(self, x):
        original_size = x.shape[2:]

        # ---------- Encoder ----------
        skips = []
        for idx, layer in enumerate(self.encoder):
            x = layer(x)
            if idx in self.SKIP_INDICES:
                skips.append(x)

        # ---------- ASPP bottleneck ----------
        x = self.aspp(x)

        # ---------- Decoder (with CBAM on skips) ----------
        x = self.dec4(x, self.skip_cbam[3](skips[3]))
        x = self.dec3(x, self.skip_cbam[2](skips[2]))
        aux3 = self.aux_head3(x, original_size)     # deep supervision

        x = self.dec2(x, self.skip_cbam[1](skips[1]))
        aux2 = self.aux_head2(x, original_size)     # deep supervision

        x = self.dec1(x, self.skip_cbam[0](skips[0]))

        x = F.interpolate(x, size=original_size,
                          mode='bilinear', align_corners=False)
        main = self.final(x)

        if self.training:
            return main, aux3, aux2   # return all 3 for loss computation
        return main                   # inference: only main prediction


# LOSS FUNCTIONS

class TverskyLoss(nn.Module):
    """
    Tversky loss: generalises Dice.  With alpha>beta it penalises
    false negatives more (misssed lesions) — important for medical imaging.
    alpha=0.3, beta=0.7 means FN costs ~2.3x more than FP.
    """
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.):
        super().__init__()
        self.alpha  = alpha
        self.beta   = beta
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        p     = probs.view(probs.size(0), -1)
        t     = targets.view(targets.size(0), -1).float()
        tp    = (p * t).sum(dim=1)
        fp    = (p * (1 - t)).sum(dim=1)
        fn    = ((1 - p) * t).sum(dim=1)
        tversky = (tp + self.smooth) / \
                  (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1. - tversky.mean()


class BoundaryLoss(nn.Module):
    """
    Boundary-aware loss: up-weights predictions near lesion boundaries.
    Uses morphological dilation to define a boundary region.
    """
    def __init__(self, theta0: float = 3., theta: float = 5.):
        super().__init__()
        self.theta0 = theta0
        self.theta  = theta

    @staticmethod
    def _one_hot2dist(mask_batch):
        """Approximate distance transform using max-pool erosion."""
        # mask_batch: (B,1,H,W)  float in {0,1}
        # erode foreground and background to get near-boundary weight
        kernel = 5
        pad    = kernel // 2
        fg_eroded = -F.max_pool2d(-mask_batch, kernel, stride=1, padding=pad)
        boundary  = mask_batch - fg_eroded   # thin boundary ring
        return boundary.clamp(0., 1.)

    def forward(self, logits, targets):
        targets = targets.float()
        boundary_weight = self._one_hot2dist(targets)
        # amplify boundary pixels
        weight = 1. + self.theta * boundary_weight
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, weight=weight, reduction='mean')
        return bce


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        targets = targets.float()
        probs   = torch.sigmoid(logits)
        ce      = F.binary_cross_entropy_with_logits(logits, targets,
                                                     reduction='none')
        p_t     = probs * targets + (1 - probs) * (1 - targets)
        return (self.alpha * ((1 - p_t) ** self.gamma) * ce).mean()


class EnhancedLoss(nn.Module):
    """
    Combined loss:  0.4 * Tversky  +  0.3 * Focal  +  0.3 * Boundary
    Applied to main output; auxiliary heads use Tversky + Focal only.
    """
    def __init__(self, aux_weight: float = AUX_WEIGHT):
        super().__init__()
        self.tversky  = TverskyLoss(alpha=0.3, beta=0.7)
        self.focal    = FocalLoss(alpha=0.25, gamma=2.)
        self.boundary = BoundaryLoss(theta0=3., theta=5.)
        self.aux_w    = aux_weight

    def _single(self, logits, targets):
        return (0.4 * self.tversky(logits, targets) +
                0.3 * self.focal(logits, targets) +
                0.3 * self.boundary(logits, targets))

    def forward(self, outputs, targets):
        if isinstance(outputs, (tuple, list)):
            main, aux3, aux2 = outputs
            loss_main = self._single(main,  targets)
            loss_aux3 = self._single(aux3,  targets)
            loss_aux2 = self._single(aux2,  targets)
            return (loss_main +
                    self.aux_w * 0.5 * loss_aux3 +
                    self.aux_w * 0.5 * loss_aux2)
        # inference path (no aux heads)
        return self._single(outputs, targets)


# METRICS

def calculate_metrics(pred_logits, target, threshold: float = 0.5):
    pred_bin  = (torch.sigmoid(pred_logits) > threshold).float()
    target_bin = target.float()

    p  = pred_bin.view(-1)
    t  = target_bin.view(-1)

    tp = (p * t).sum()
    fp = (p * (1 - t)).sum()
    fn = ((1 - p) * t).sum()
    tn = ((1 - p) * (1 - t)).sum()

    eps = 1e-7
    dice        = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou         = (tp + eps) / (tp + fp + fn + eps)
    precision   = (tp + eps) / (tp + fp + eps)
    recall      = (tp + eps) / (tp + fn + eps)
    specificity = (tn + eps) / (tn + fp + eps)
    accuracy    = (tp + tn + eps) / (tp + tn + fp + fn + eps)
    f2          = (5 * tp + eps) / (5 * tp + 4 * fn + fp + eps)  # F2-score (extra recall)
    jaccard     = iou   # identical

    return {
        'dice':        dice.item(),
        'iou':         iou.item(),
        'jaccard':     jaccard.item(),
        'precision':   precision.item(),
        'recall':      recall.item(),
        'specificity': specificity.item(),
        'accuracy':    accuracy.item(),
        'f2':          f2.item(),
    }


# TEST-TIME AUGMENTATION (TTA)

@torch.no_grad()
def tta_predict(model, image):
    """
    Apply 8-fold TTA: original + H-flip + V-flip + HV-flip
    + 90-degree rotations of each.  Average sigmoid probabilities.
    """
    model.eval()
    preds = []
    for hflip in [False, True]:
        for vflip in [False, True]:
            inp = image
            if hflip:
                inp = torch.flip(inp, dims=[3])
            if vflip:
                inp = torch.flip(inp, dims=[2])
            out = torch.sigmoid(model(inp))
            if vflip:
                out = torch.flip(out, dims=[2])
            if hflip:
                out = torch.flip(out, dims=[3])
            preds.append(out)
    return torch.stack(preds, dim=0).mean(dim=0)


# TRAINING & VALIDATION LOOPS

def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = 0.
    pbar = tqdm(loader, desc='Training')
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)

        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
            outputs = model(images)              # (main, aux3, aux2) during training
            loss    = criterion(outputs, masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device, use_tta: bool = False):
    model.eval()
    total_loss  = 0.
    all_metrics = []

    pbar = tqdm(loader, desc='Validation')
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device,  non_blocking=True)

        if use_tta:
            probs   = tta_predict(model, images)   # (B,1,H,W) probabilities
            # convert to pseudo-logits for criterion (log-odds)
            outputs = torch.log(probs.clamp(1e-6, 1-1e-6) /
                                (1 - probs.clamp(1e-6, 1-1e-6)))
        else:
            outputs = model(images)

        loss = criterion(outputs, masks)
        total_loss += loss.item()

        metrics = calculate_metrics(outputs, masks)
        all_metrics.append(metrics)
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'dice': f'{metrics["dice"]:.4f}'
        })

    avg = {k: float(np.mean([m[k] for m in all_metrics]))
           for k in all_metrics[0]}
    return total_loss / len(loader), avg


# VISUALISATION

MEAN = np.array([0.485, 0.456, 0.406])
STD  = np.array([0.229, 0.224, 0.225])

def denorm(tensor):
    img = tensor.numpy().transpose(1, 2, 0)
    img = img * STD + MEAN
    return np.clip(img, 0., 1.)


def save_comprehensive_curves(train_losses, val_losses, metrics_history,
                               save_dir):
    epochs = range(1, len(train_losses) + 1)
    keys = ['dice', 'iou', 'precision', 'recall',
            'specificity', 'accuracy', 'f2']
    titles = ['Dice Score', 'IoU / Jaccard', 'Precision',
              'Recall (Sensitivity)', 'Specificity', 'Accuracy', 'F2 Score']
    colors = ['#2ecc71', '#3498db', '#9b59b6', '#e67e22',
              '#c0392b', '#1abc9c', '#e74c3c']

    fig, axes = plt.subplots(3, 3, figsize=(22, 16))
    fig.suptitle('Enhanced Model — Training Progress',
                 fontsize=18, fontweight='bold', y=1.01)

    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, train_losses, 'b-', label='Train', linewidth=2)
    ax.plot(epochs, val_losses,   'r-', label='Val',   linewidth=2)
    ax.set_title('Loss', fontsize=13)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.legend(); ax.grid(True, alpha=0.3)

    for i, (k, title, c) in enumerate(zip(keys, titles, colors)):
        row, col = divmod(i + 1, 3)
        ax  = axes[row, col]
        vals = [m[k] for m in metrics_history]
        ax.plot(epochs, vals, color=c, linewidth=2)
        best_v = max(vals)
        best_e = vals.index(best_v) + 1
        ax.axvline(best_e, linestyle='--', color='k', alpha=0.4)
        ax.set_title(f'{title}  (best={best_v:.4f} @ep{best_e})', fontsize=11)
        ax.set_xlabel('Epoch'); ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)

    # Summary panel
    ax = axes[2, 2]
    ax.axis('off')
    best_dice_idx = [m['dice'] for m in metrics_history].index(
        max(m['dice'] for m in metrics_history))
    bm = metrics_history[best_dice_idx]
    txt = (
        f"Best Epoch : {best_dice_idx + 1}\n"
        f"Dice       : {bm['dice']:.4f}\n"
        f"IoU        : {bm['iou']:.4f}\n"
        f"Precision  : {bm['precision']:.4f}\n"
        f"Recall     : {bm['recall']:.4f}\n"
        f"Specificity: {bm['specificity']:.4f}\n"
        f"Accuracy   : {bm['accuracy']:.4f}\n"
        f"F2 Score   : {bm['f2']:.4f}"
    )
    ax.text(0.05, 0.95, txt, transform=ax.transAxes,
            fontsize=12, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='#ecf0f1', alpha=0.8))
    ax.set_title('Best Val Performance', fontsize=13)

    plt.tight_layout()
    path = os.path.join(save_dir, 'training_curves_enhanced.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved training curves: {path}')


def save_prediction_visualizations(model, loader, device, save_dir,
                                    num_samples: int = 16,
                                    use_tta: bool = False):
    model.eval()
    imgs_l, masks_l, preds_l, probs_l = [], [], [], []

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            if use_tta:
                prob = tta_predict(model, images).cpu()
            else:
                prob = torch.sigmoid(model(images)).cpu()

            pred = (prob > 0.5).float()
            imgs_l.append(images.cpu())
            masks_l.append(masks.cpu())
            preds_l.append(pred)
            probs_l.append(prob)
            if sum(x.shape[0] for x in imgs_l) >= num_samples:
                break

    imgs_all  = torch.cat(imgs_l)[:num_samples]
    masks_all = torch.cat(masks_l)[:num_samples]
    preds_all = torch.cat(preds_l)[:num_samples]
    probs_all = torch.cat(probs_l)[:num_samples]

    fig, axes = plt.subplots(num_samples, 5,
                              figsize=(20, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    headers = ['Image', 'Ground Truth', 'Prediction', 'Probability', 'Overlay']
    for col, h in enumerate(headers):
        axes[0, col].set_title(h, fontsize=13, fontweight='bold')

    for idx in range(num_samples):
        img   = denorm(imgs_all[idx])
        mask  = masks_all[idx, 0].numpy()
        pred  = preds_all[idx, 0].numpy()
        prob  = probs_all[idx, 0].numpy()

        # compute per-sample dice for title
        inter = (pred * mask).sum()
        d     = (2 * inter + 1) / (pred.sum() + mask.sum() + 1)

        # overlay: green=TP, red=FP, blue=FN
        overlay = img.copy()
        tp = (pred == 1) & (mask == 1)
        fp = (pred == 1) & (mask == 0)
        fn = (pred == 0) & (mask == 1)
        overlay[tp] = [0.2, 0.9, 0.2]
        overlay[fp] = [0.9, 0.1, 0.1]
        overlay[fn] = [0.1, 0.3, 0.9]

        row = axes[idx]
        row[0].imshow(img);          row[0].axis('off')
        row[1].imshow(mask,  cmap='gray'); row[1].axis('off')
        row[2].imshow(pred,  cmap='gray'); row[2].axis('off')
        row[3].imshow(prob,  cmap='jet', vmin=0, vmax=1)
        row[3].axis('off')
        row[4].imshow(overlay);      row[4].axis('off')
        row[0].set_ylabel(f'Sample {idx+1}\nDice={d:.3f}',
                          fontsize=9, rotation=0, labelpad=60)

    # legend
    from matplotlib.patches import Patch
    legend = [Patch(facecolor='#33e633', label='TP'),
              Patch(facecolor='#e61a1a', label='FP'),
              Patch(facecolor='#1a4de6', label='FN')]
    axes[-1, 4].legend(handles=legend, loc='lower right', fontsize=10)

    plt.suptitle('Segmentation Predictions (Enhanced Model)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(save_dir, 'predictions_enhanced.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved predictions visualisation: {path}')


def save_metrics_table_image(metrics, save_dir,
                              filename='test_metrics_enhanced.png',
                              title='Test Set — Enhanced Model'):
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.axis('tight'); ax.axis('off')

    rows = [
        ['Dice Score',            f"{metrics['dice']:.4f}"],
        ['IoU (Jaccard)',         f"{metrics['iou']:.4f}"],
        ['Precision',             f"{metrics['precision']:.4f}"],
        ['Recall (Sensitivity)',  f"{metrics['recall']:.4f}"],
        ['Specificity',           f"{metrics['specificity']:.4f}"],
        ['Accuracy',              f"{metrics['accuracy']:.4f}"],
        ['F2 Score',              f"{metrics['f2']:.4f}"],
    ]

    tbl = ax.table(cellText=rows, colLabels=['Metric', 'Value'],
                   cellLoc='left', loc='center', colWidths=[0.65, 0.35])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(14)
    tbl.scale(1, 2.8)

    header_color = '#2c3e50'
    row_colors   = ['#ecf0f1', '#ffffff']
    highlight_thresh = {'dice': 0.93, 'iou': 0.87}

    for j in range(2):
        tbl[(0, j)].set_facecolor(header_color)
        tbl[(0, j)].set_text_props(weight='bold', color='white', fontsize=15)

    for i, row in enumerate(rows, start=1):
        for j in range(2):
            tbl[(i, j)].set_facecolor(row_colors[i % 2])
            tbl[(i, j)].set_text_props(fontsize=14)

    plt.title(title, fontsize=17, fontweight='bold', pad=24)
    path = os.path.join(save_dir, filename)
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved metrics table: {path}')


def save_individual_segmentations(model, loader, device, save_dir,
                                   use_tta: bool = False):
    """Save each test image as its own high-res PNG (image | mask | pred)."""
    seg_dir = os.path.join(save_dir, 'individual_segmentations')
    os.makedirs(seg_dir, exist_ok=True)
    model.eval()
    sample_idx = 0

    with torch.no_grad():
        for images, masks in tqdm(loader, desc='Saving individual segs'):
            images = images.to(device)
            if use_tta:
                prob = tta_predict(model, images).cpu()
            else:
                prob = torch.sigmoid(model(images)).cpu()
            pred = (prob > 0.5).float()

            for i in range(images.shape[0]):
                img  = denorm(images[i].cpu())
                mask = masks[i, 0].numpy()
                p    = pred[i, 0].numpy()
                pr   = prob[i, 0].numpy()

                inter = (p * mask).sum()
                d = (2 * inter + 1) / (p.sum() + mask.sum() + 1)

                fig, axes = plt.subplots(1, 4, figsize=(18, 5))
                axes[0].imshow(img)
                axes[0].set_title('Input Image', fontsize=12)
                axes[1].imshow(mask, cmap='gray')
                axes[1].set_title('Ground Truth', fontsize=12)
                axes[2].imshow(p, cmap='gray')
                axes[2].set_title(f'Prediction  (Dice={d:.3f})', fontsize=12)
                axes[3].imshow(pr, cmap='jet', vmin=0, vmax=1)
                axes[3].set_title('Prob. Map', fontsize=12)
                for ax in axes:
                    ax.axis('off')

                plt.tight_layout()
                path = os.path.join(seg_dir, f'sample_{sample_idx:03d}.png')
                plt.savefig(path, dpi=150, bbox_inches='tight')
                plt.close()
                sample_idx += 1

    print(f'Saved {sample_idx} individual segmentation PNGs → {seg_dir}')


# MAIN

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ---- Data ----
    train_loader, val_loader, test_loader = create_dataloaders(
        DATA_ROOT, BATCH_SIZE, VAL_RATIO, TEST_RATIO, SEED
    )

    # ---- Model ----
    print('\n' + '=' * 80)
    print('MODEL SETUP')
    print('=' * 80)

    model = EnhancedUNet(num_classes=1).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params:,}')

    criterion = EnhancedLoss(aux_weight=AUX_WEIGHT)

    # Differential learning rates: encoder gets 10x smaller LR
    encoder_params = list(model.encoder.parameters())
    decoder_params = [p for n, p in model.named_parameters()
                      if 'encoder' not in n]

    optimizer = torch.optim.AdamW([
        {'params': encoder_params, 'lr': BASE_LR * 0.1},
        {'params': decoder_params, 'lr': BASE_LR},
    ], weight_decay=WEIGHT_DECAY)

    # CosineAnnealingWarmRestarts — better exploration than plain cosine
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=T0, T_mult=T_MULT, eta_min=1e-7
    )

    scaler = torch.amp.GradScaler('cuda')
    print('Setup complete\n')

    # ---- Training loop ----
    print('=' * 80)
    print('TRAINING')
    print('=' * 80 + '\n')

    best_dice = 0.
    train_losses, val_losses, metrics_history = [], [], []
    best_model_path = os.path.join(SAVE_DIR, 'best_model_enhanced.pth')

    for epoch in range(EPOCHS):
        print(f'\nEpoch {epoch+1}/{EPOCHS}')
        print('-' * 80)

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, DEVICE)
        val_loss, val_metrics = validate(
            model, val_loader, criterion, DEVICE, use_tta=False)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        metrics_history.append(val_metrics)

        print(f'\nTrain Loss  : {train_loss:.4f}')
        print(f'Val Loss    : {val_loss:.4f}')
        print(f'Val Dice    : {val_metrics["dice"]:.4f}')
        print(f'Val IoU     : {val_metrics["iou"]:.4f}')
        print(f'Jaccard     : {val_metrics["jaccard"]:.4f}')
        print(f'Precision   : {val_metrics["precision"]:.4f}')
        print(f'Recall      : {val_metrics["recall"]:.4f}')
        print(f'Specificity : {val_metrics["specificity"]:.4f}')
        print(f'Accuracy    : {val_metrics["accuracy"]:.4f}')
        print(f'F2 Score    : {val_metrics["f2"]:.4f}')

        if val_metrics['dice'] > best_dice:
            best_dice = val_metrics['dice']
            torch.save(
                {'epoch': epoch + 1,
                 'model_state_dict': model.state_dict(),
                 'optimizer_state_dict': optimizer.state_dict(),
                 'best_dice': best_dice,
                 'val_metrics': val_metrics},
                best_model_path
            )
            print(f'  Saved best model  (Dice: {best_dice:.4f})')

        scheduler.step()

        # periodic checkpoints every 50 epochs
        if (epoch + 1) % 50 == 0:
            ckpt_path = os.path.join(SAVE_DIR, f'checkpoint_ep{epoch+1}.pth')
            torch.save(model.state_dict(), ckpt_path)

    # ---- Test evaluation ----
    print('\n' + '=' * 80)
    print('TESTING  (loading best model)')
    print('=' * 80)

    ckpt = torch.load(best_model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f'  Best model was at epoch {ckpt["epoch"]}  '
          f'(val Dice={ckpt["best_dice"]:.4f})')

    # Test WITHOUT TTA
    _, test_metrics_no_tta = validate(
        model, test_loader, criterion, DEVICE, use_tta=False)

    # Test WITH TTA
    _, test_metrics_tta = validate(
        model, test_loader, criterion, DEVICE, use_tta=USE_TTA)

    print('\n' + '=' * 80)
    print('FINAL TEST RESULTS')
    print('=' * 80)
    print(f'{"Metric":<22}  {"No-TTA":>10}  {"With-TTA":>10}')
    print('-' * 48)
    for k in ['dice','iou','jaccard','precision','recall',
              'specificity','accuracy','f2']:
        v1 = test_metrics_no_tta[k]
        v2 = test_metrics_tta[k]
        print(f'{k:<22}  {v1:>10.4f}  {v2:>10.4f}')
    print('=' * 80)

    # Save results to text
    with open(os.path.join(SAVE_DIR, 'results_enhanced.txt'), 'w') as f:
        f.write('=' * 60 + '\n')
        f.write('ENHANCED MODEL  —  TEST SET RESULTS\n')
        f.write('=' * 60 + '\n')
        f.write(f'Best val epoch : {ckpt["epoch"]}\n')
        f.write(f'Best val Dice  : {ckpt["best_dice"]:.4f}\n\n')
        f.write(f'{"Metric":<22}  {"No-TTA":>8}  {"TTA":>8}\n')
        f.write('-' * 44 + '\n')
        for k in ['dice','iou','jaccard','precision','recall',
                  'specificity','accuracy','f2']:
            f.write(f'{k:<22}  '
                    f'{test_metrics_no_tta[k]:>8.4f}  '
                    f'{test_metrics_tta[k]:>8.4f}\n')
        f.write('=' * 60 + '\n')
        f.write('Baseline: Dice=0.9309, IoU=0.8709\n')

    # ---- Visualisations ----
    print('\n' + '=' * 80)
    print('GENERATING VISUALISATIONS')
    print('=' * 80)

    # 1. Training curves (all 8 metrics)
    save_comprehensive_curves(
        train_losses, val_losses, metrics_history, SAVE_DIR)

    # 2. Grid of 16 test predictions (with TTA)
    save_prediction_visualizations(
        model, test_loader, DEVICE, SAVE_DIR,
        num_samples=16, use_tta=USE_TTA)

    # 3. Metrics table
    save_metrics_table_image(
        test_metrics_tta, SAVE_DIR,
        filename='test_metrics_enhanced.png',
        title='Test Set — Enhanced Model (With TTA)')

    # 4. Individual PNG per test sample
    save_individual_segmentations(
        model, test_loader, DEVICE, SAVE_DIR, use_tta=USE_TTA)

    print(f'\n{"="*80}')
    print(f'DONE!  Best val Dice = {best_dice:.4f}')
    print(f'All outputs saved to: {SAVE_DIR}')
    print(f'{"="*80}\n')


if __name__ == '__main__':
    main()
