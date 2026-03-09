"""
Fine-tune MobileNetV2 on COCO crops for reCAPTCHA category classification.

Downloads COCO annotations, extracts bounding box crops for target categories,
fine-tunes MobileNetV2's classifier head, and exports to ONNX.

Usage:
    python training/train.py
"""

import json
import os
import random
import shutil
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# ─── Config ───────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent / "data"
CROPS_DIR = DATA_DIR / "crops"
MODEL_OUT = Path(__file__).parent.parent / "src" / "captcha_solver" / "captcha_mobilenet.onnx"
RELEASE_MODEL = Path(__file__).parent / "captcha_mobilenet.onnx"

COCO_ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_IMG_URL_TEMPLATE = "http://images.cocodataset.org/train2017/{:012d}.jpg"
COCO_VAL_IMG_URL_TEMPLATE = "http://images.cocodataset.org/val2017/{:012d}.jpg"

# COCO category IDs → our class names
COCO_CATEGORIES = {
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic_light",
    11: "fire_hydrant",
    14: "parking_meter",
}

# Our full class list (what the model outputs)
# Index 0 = background/other, rest = reCAPTCHA categories
CLASS_NAMES = [
    "background",       # 0
    "traffic_light",    # 1
    "bus",              # 2
    "bicycle",          # 3
    "motorcycle",       # 4
    "car",              # 5
    "bridge",           # 6
    "boat",             # 7
    "airplane",         # 8
    "train",            # 9
    "truck",            # 10
    "fire_hydrant",     # 11
    "parking_meter",    # 12
    "mountain",         # 13
    "tractor",          # 14
    "crosswalk",        # 15
    "stair",            # 16
    "palm_tree",        # 17
    "chimney",          # 18
]

NUM_CLASSES = len(CLASS_NAMES)

# Map COCO category name → our class index
COCO_TO_CLASS = {}
for coco_id, coco_name in COCO_CATEGORIES.items():
    if coco_name in CLASS_NAMES:
        COCO_TO_CLASS[coco_id] = CLASS_NAMES.index(coco_name)

# Training params
BATCH_SIZE = 32
NUM_EPOCHS = 15
LR = 0.001
MAX_SAMPLES_PER_CLASS = 800
MIN_CROP_SIZE = 32  # skip tiny bounding boxes
IMG_SIZE = 224

# ─── Download & Extract COCO ─────────────────────────────────────────────────


def download_coco_annotations():
    """Download and extract COCO 2017 annotations."""
    ann_dir = DATA_DIR / "annotations"
    train_ann = ann_dir / "instances_train2017.json"
    val_ann = ann_dir / "instances_val2017.json"

    if train_ann.exists() and val_ann.exists():
        print("COCO annotations already downloaded")
        return train_ann, val_ann

    zip_path = DATA_DIR / "annotations_trainval2017.zip"
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not zip_path.exists():
        print(f"Downloading COCO annotations ({COCO_ANN_URL})...")
        req = urllib.request.Request(COCO_ANN_URL, headers={"User-Agent": "captcha-solver-trainer/1.0"})
        with urllib.request.urlopen(req, timeout=300) as resp:
            with open(zip_path, "wb") as f:
                shutil.copyfileobj(resp, f)
        print(f"Downloaded {zip_path.stat().st_size / 1e6:.0f} MB")

    print("Extracting annotations...")
    import zipfile
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(DATA_DIR)
    print("Done")

    return train_ann, val_ann


def parse_coco_annotations(ann_path):
    """Parse COCO annotations and return relevant entries."""
    print(f"Parsing {ann_path.name}...")
    with open(ann_path) as f:
        data = json.load(f)

    # Build image lookup
    images = {img["id"]: img for img in data["images"]}

    # Filter annotations for our target categories
    relevant = []
    for ann in data["annotations"]:
        cat_id = ann["category_id"]
        if cat_id in COCO_TO_CLASS and ann["iscrowd"] == 0:
            bbox = ann["bbox"]  # [x, y, w, h]
            if bbox[2] >= MIN_CROP_SIZE and bbox[3] >= MIN_CROP_SIZE:
                img_info = images[ann["image_id"]]
                relevant.append({
                    "image_id": ann["image_id"],
                    "file_name": img_info["file_name"],
                    "bbox": bbox,
                    "class_idx": COCO_TO_CLASS[cat_id],
                    "class_name": CLASS_NAMES[COCO_TO_CLASS[cat_id]],
                })

    print(f"  Found {len(relevant)} relevant annotations")
    return relevant


def download_image(image_id, file_name, is_val=False):
    """Download a single COCO image if not cached."""
    img_dir = DATA_DIR / ("val2017" if is_val else "train2017")
    img_dir.mkdir(parents=True, exist_ok=True)
    img_path = img_dir / file_name

    if img_path.exists():
        return img_path

    if is_val:
        url = COCO_VAL_IMG_URL_TEMPLATE.format(image_id)
    else:
        url = COCO_IMG_URL_TEMPLATE.format(image_id)

    req = urllib.request.Request(url, headers={"User-Agent": "captcha-solver-trainer/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            with open(img_path, "wb") as f:
                shutil.copyfileobj(resp, f)
    except Exception as e:
        print(f"  Failed to download {file_name}: {e}")
        return None

    return img_path


def extract_crops(annotations, is_val=False):
    """Download images and extract bounding box crops."""
    # Balance classes
    by_class = {}
    for ann in annotations:
        cls = ann["class_idx"]
        by_class.setdefault(cls, []).append(ann)

    print("\nClass distribution in annotations:")
    for cls_idx, anns in sorted(by_class.items()):
        print(f"  {CLASS_NAMES[cls_idx]:20s}: {len(anns)} annotations")

    # Sample up to MAX_SAMPLES_PER_CLASS per class
    selected = []
    for cls_idx, anns in by_class.items():
        if len(anns) > MAX_SAMPLES_PER_CLASS:
            anns = random.sample(anns, MAX_SAMPLES_PER_CLASS)
        selected.extend(anns)

    # Get unique images we need to download
    needed_images = {}
    for ann in selected:
        img_id = ann["image_id"]
        if img_id not in needed_images:
            needed_images[img_id] = ann["file_name"]

    print(f"\nNeed {len(needed_images)} unique images for {len(selected)} crops")
    print("Downloading images...")

    # Download and crop
    crops_saved = 0
    failed = 0
    for i, ann in enumerate(selected):
        cls_name = ann["class_name"]
        cls_dir = CROPS_DIR / cls_name
        cls_dir.mkdir(parents=True, exist_ok=True)

        crop_path = cls_dir / f"{ann['image_id']}_{int(ann['bbox'][0])}_{int(ann['bbox'][1])}.jpg"
        if crop_path.exists():
            crops_saved += 1
            continue

        img_path = download_image(ann["image_id"], ann["file_name"], is_val)
        if img_path is None:
            failed += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            failed += 1
            continue

        x, y, w, h = [int(v) for v in ann["bbox"]]
        # Add 10% padding for context
        pad_x = int(w * 0.1)
        pad_y = int(h * 0.1)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(img.shape[1], x + w + pad_x)
        y2 = min(img.shape[0], y + h + pad_y)

        crop = img[y1:y2, x1:x2]
        if crop.shape[0] < 10 or crop.shape[1] < 10:
            failed += 1
            continue

        cv2.imwrite(str(crop_path), crop)
        crops_saved += 1

        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(selected)}] {crops_saved} crops saved, {failed} failed")

    print(f"\nTotal: {crops_saved} crops saved, {failed} failed")
    return crops_saved


def download_open_images_crops(category, class_idx, max_samples=500):
    """Download crops from Open Images for categories not in COCO.

    Uses the OID CSV files to find image IDs, then downloads crops.
    For simplicity, we use web search to find and download sample images.
    """
    cls_dir = CROPS_DIR / CLASS_NAMES[class_idx]
    if cls_dir.exists() and len(list(cls_dir.glob("*.jpg"))) >= 50:
        print(f"  {category}: already have {len(list(cls_dir.glob('*.jpg')))} crops")
        return

    cls_dir.mkdir(parents=True, exist_ok=True)
    # For non-COCO categories, we'll generate synthetic negative examples
    # and rely on the "background" class for now.
    # In production, you'd use Open Images V7 or manual collection.
    print(f"  {category}: no COCO data available (will use background class for negatives)")


# ─── Background/Negative Class ──────────────────────────────────────────────


def create_background_crops(all_annotations, num_samples=800):
    """Create background crops from COCO images that DON'T contain target objects."""
    bg_dir = CROPS_DIR / "background"
    if bg_dir.exists() and len(list(bg_dir.glob("*.jpg"))) >= num_samples // 2:
        print(f"Background crops: already have {len(list(bg_dir.glob('*.jpg')))}")
        return

    bg_dir.mkdir(parents=True, exist_ok=True)

    # Get IDs of images that contain our target categories
    target_image_ids = {ann["image_id"] for ann in all_annotations}

    # Load the full annotation file to find images WITHOUT our targets
    ann_path = DATA_DIR / "annotations" / "instances_val2017.json"
    with open(ann_path) as f:
        data = json.load(f)

    # Find images that don't contain any target category
    non_target_images = []
    for img in data["images"]:
        if img["id"] not in target_image_ids:
            non_target_images.append(img)

    random.shuffle(non_target_images)
    saved = 0

    print(f"Creating background crops from {len(non_target_images)} non-target images...")
    for img_info in non_target_images[:num_samples]:
        img_path = download_image(img_info["id"], img_info["file_name"], is_val=True)
        if img_path is None:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        # Random crop from the image
        crop_size = min(h, w, max(h // 3, 100))
        cx = random.randint(0, w - crop_size)
        cy = random.randint(0, h - crop_size)
        crop = img[cy:cy + crop_size, cx:cx + crop_size]

        crop_path = bg_dir / f"bg_{img_info['id']}_{cx}_{cy}.jpg"
        cv2.imwrite(str(crop_path), crop)
        saved += 1

        if saved >= num_samples:
            break

    print(f"  Saved {saved} background crops")


# ─── Dataset ─────────────────────────────────────────────────────────────────


class CropDataset(Dataset):
    """Dataset of COCO bounding box crops for training."""

    def __init__(self, root_dir, transform=None, max_per_class=None):
        self.samples = []
        self.transform = transform

        for cls_idx, cls_name in enumerate(CLASS_NAMES):
            cls_dir = Path(root_dir) / cls_name
            if not cls_dir.exists():
                continue
            files = sorted(cls_dir.glob("*.jpg"))
            if max_per_class and len(files) > max_per_class:
                random.shuffle(files)
                files = files[:max_per_class]
            for f in files:
                self.samples.append((str(f), cls_idx))

        random.shuffle(self.samples)
        print(f"Dataset: {len(self.samples)} total samples across {NUM_CLASSES} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        if img is None:
            # Return a blank image if read fails
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        return img, label


# ─── Training ────────────────────────────────────────────────────────────────


def build_model():
    """Build MobileNetV2 with custom classifier head."""
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    # Freeze backbone
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace classifier
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(model.last_channel, NUM_CLASSES),
    )

    return model


def train_model(model, train_loader, val_loader, device):
    """Train the model."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    best_acc = 0.0
    best_state = None

    for epoch in range(NUM_EPOCHS):
        # Train
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        train_acc = 100. * correct / total
        train_loss = running_loss / len(train_loader)

        # Validate
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100. * val_correct / val_total if val_total > 0 else 0

        print(
            f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
            f"Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.1f}% | "
            f"Val Acc: {val_acc:.1f}%"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        scheduler.step()

    if best_state:
        model.load_state_dict(best_state)
        print(f"\nBest validation accuracy: {best_acc:.1f}%")
        # Save checkpoint so we don't lose training if export fails
        ckpt_path = Path(__file__).parent / "best_model.pth"
        torch.save(best_state, ckpt_path)
        print(f"Checkpoint saved to {ckpt_path}")

    return model


def export_onnx(model, device):
    """Export trained model to ONNX format."""
    model.eval()
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE).to(device)

    # Export
    torch.onnx.export(
        model,
        dummy,
        str(RELEASE_MODEL),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=12,
    )

    size_mb = RELEASE_MODEL.stat().st_size / (1024 * 1024)
    print(f"\nExported to {RELEASE_MODEL} ({size_mb:.1f} MB)")

    # Also save class names alongside the model
    meta_path = RELEASE_MODEL.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump({"classes": CLASS_NAMES, "input_size": IMG_SIZE}, f, indent=2)
    print(f"Class metadata saved to {meta_path}")

    return RELEASE_MODEL


# ─── Main ────────────────────────────────────────────────────────────────────


def main():
    random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Classes: {NUM_CLASSES} ({', '.join(CLASS_NAMES)})")

    # Step 1: Download COCO annotations
    print("\n" + "=" * 60)
    print("STEP 1: Download COCO annotations")
    print("=" * 60)
    train_ann_path, val_ann_path = download_coco_annotations()

    # Step 2: Parse annotations
    print("\n" + "=" * 60)
    print("STEP 2: Parse annotations")
    print("=" * 60)
    train_anns = parse_coco_annotations(train_ann_path)
    val_anns = parse_coco_annotations(val_ann_path)

    # Step 3: Extract crops
    print("\n" + "=" * 60)
    print("STEP 3: Extract crops from COCO images")
    print("=" * 60)
    print("\n--- Train crops ---")
    extract_crops(train_anns, is_val=False)
    print("\n--- Val crops ---")
    extract_crops(val_anns, is_val=True)

    # Step 4: Background crops
    print("\n" + "=" * 60)
    print("STEP 4: Create background/negative crops")
    print("=" * 60)
    create_background_crops(val_anns)

    # Step 5: Non-COCO categories
    print("\n" + "=" * 60)
    print("STEP 5: Non-COCO categories")
    print("=" * 60)
    non_coco = ["bridge", "mountain", "tractor", "crosswalk", "stair", "palm_tree", "chimney"]
    for cat in non_coco:
        cls_idx = CLASS_NAMES.index(cat)
        download_open_images_crops(cat, cls_idx)

    # Step 6: Build datasets
    print("\n" + "=" * 60)
    print("STEP 6: Build datasets & train")
    print("=" * 60)

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = CropDataset(CROPS_DIR, transform=train_transform, max_per_class=MAX_SAMPLES_PER_CLASS)
    val_dataset = CropDataset(CROPS_DIR, transform=val_transform, max_per_class=200)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Step 7: Train
    model = build_model().to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    model = train_model(model, train_loader, val_loader, device)

    # Step 8: Export
    print("\n" + "=" * 60)
    print("STEP 8: Export to ONNX")
    print("=" * 60)
    export_onnx(model, device)

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
