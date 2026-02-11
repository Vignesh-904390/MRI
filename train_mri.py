import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split

# ============================
# PATHS & DEVICE
# ============================
DATA_DIR = "data - 3/MRI DATA"   # <-- CHANGE THIS PATH
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "mri_effnet_best.pth")
os.makedirs(MODEL_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("📦 Using device:", device)

# ============================
# DATA TRANSFORMS (MRI-SAFE)
# ============================
train_tf = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ============================
# LOAD DATASET
# ============================
print("📁 Loading MRI dataset...")
full_dataset = datasets.ImageFolder(DATA_DIR, transform=train_tf)
num_classes = len(full_dataset.classes)

# ============================
# TRAIN / VALIDATION SPLIT
# ============================
train_size = int(0.9 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

print(f"✅ Total: {len(full_dataset)} | Train: {train_size} | Val: {val_size}")
print("🧪 Classes:", full_dataset.classes)

# ============================
# MIXUP FUNCTION
# ============================
def mixup(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam

# ============================
# LOAD EFFICIENTNET-B0
# ============================
print("🧠 Loading EfficientNet-B0...")
model = models.efficientnet_b0(pretrained=True)

# Fine-tune last blocks only
for name, param in model.named_parameters():
    param.requires_grad = "features.6" in name or "features.7" in name

model.classifier[1] = nn.Linear(
    model.classifier[1].in_features, num_classes
)
model = model.to(device)

# ============================
# LOSS, OPTIMIZER, SCHEDULER
# ============================
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = torch.optim.AdamW(
    model.parameters(), lr=0.0004, weight_decay=0.01
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=5, T_mult=1
)

# ============================
# TRAINING LOOP
# ============================
best_val_loss = float("inf")
EPOCHS = 2000

print("🚀 MRI Training Started...\n")

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        imgs, y_a, y_b, lam = mixup(imgs, labels)

        optimizer.zero_grad()
        outputs = model(imgs)

        loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()

        train_loss += loss.item()

    scheduler.step()

    # ============================
    # VALIDATION
    # ============================
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            val_loss += criterion(outputs, labels).item()

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f}"
    )

    # ============================
    # SAVE BEST MODEL
    # ============================
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save({
            "model_state_dict": model.state_dict(),
            "classes": full_dataset.classes
        }, MODEL_PATH)
        print("💾 Best MRI model saved ✔")

print("\n🎉 MRI Training Finished")
print("📌 BEST MODEL:", MODEL_PATH)