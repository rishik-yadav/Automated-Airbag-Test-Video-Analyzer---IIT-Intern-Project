import os
import glob
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import segmentation_models_pytorch as smp

# --- 1. CONFIGURATION & INDIVIDUAL FILE SPLIT ---
random.seed(42)
MASK_DIR = r"dataset\masks"
IMG_DIR = r"dataset\images"

mask_files = glob.glob(os.path.join(MASK_DIR, "*.png"))
bases = [os.path.splitext(os.path.basename(m))[0] for m in mask_files]

# Shuffle all individual frames directly
random.shuffle(bases)
n_frames = len(bases)

# Distribute 70% Train / 15% Val / 15% Test
train_idx = int(0.70 * n_frames)
val_idx = int(0.85 * n_frames)

train_files = bases[:train_idx]
val_files = bases[train_idx:val_idx]
test_files = bases[val_idx:]

print(f"Total frame assets found: {n_frames}")
print(f"Frames mapped -> Train: {len(train_files)} | Val: {len(val_files)} | Test: {len(test_files)}")

# --- 2. DATASET CLASS ---
class AirbagDataset(Dataset):
    def __init__(self, img_dir, mask_dir, files, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.files = files
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        
        # Handle variations in raw image file extensions safely
        img_path = os.path.join(self.img_dir, name + ".png")
        if not os.path.exists(img_path):
            img_path = os.path.join(self.img_dir, name + ".jpg")
            
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        mask = cv2.imread(os.path.join(self.mask_dir, name + ".png"), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype("float32") # Convert to binary threshold array
        
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img, mask = augmented["image"], augmented["mask"]
            
        # Format tensors to PyTorch structure: Channels First (C, H, W)
        img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        mask = torch.from_numpy(mask).unsqueeze(0).float()
        return img, mask

# --- 3. AUGMENTATIONS ---
train_transforms = A.Compose([
    A.Resize(512, 512),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
    A.GaussNoise(p=0.2) # Mimics development gas/smoke artifacts
])

val_transforms = A.Compose([
    A.Resize(512, 512)
])

# --- 4. DATA LOADERS ---
train_ds = AirbagDataset(IMG_DIR, MASK_DIR, train_files, train_transforms)
val_ds = AirbagDataset(IMG_DIR, MASK_DIR, val_files, val_transforms)

train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0, drop_last=True)
val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=0)

# --- 5. INITIALIZE MODEL & HYBRID LOSS ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using computing hardware device: {device.upper()}")

model = smp.DeepLabV3Plus(
    encoder_name="resnet50", 
    encoder_weights="imagenet", 
    in_channels=3, 
    classes=1, 
    activation=None
)
model.to(device)

dice_loss = smp.losses.DiceLoss(mode="binary")
bce_loss = torch.nn.BCEWithLogitsLoss()

def hybrid_criterion(logits, targets):
    return 0.5 * dice_loss(logits, targets) + 0.5 * bce_loss(logits, targets)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

# --- 6. TRAINING ENGINE LOOP ---
best_val_loss = float("inf")

print("Starting DeepLabV3+ Model Training Pipeline...")
for epoch in range(20): # Adjust total epochs as required by evaluation metrics
    model.train()
    running_train_loss = 0.0
    
    for imgs, masks in train_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = hybrid_criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        running_train_loss += loss.item() * imgs.size(0)
        
    # Run validation checks at the end of every epoch
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)
            loss = hybrid_criterion(outputs, masks)
            running_val_loss += loss.item() * imgs.size(0)
            
    epoch_train = running_train_loss / len(train_ds)
    epoch_val = running_val_loss / len(val_ds)
    
    print(f"Epoch {epoch+1:02d} | Train Loss: {epoch_train:.4f} | Val Loss: {epoch_val:.4f}")
    
    # Save the absolute best model weights based on validation loss performance
    if epoch_val < best_val_loss:
        best_val_loss = epoch_val
        torch.save(model.state_dict(), "deeplabv3plus_airbag_best.pt")
        print("--> Saved updated checkpoint matching lower validation loss!")

print("Training cycle successfully complete!")