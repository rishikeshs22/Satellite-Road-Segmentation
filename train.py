import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import os
import torchvision.transforms.functional

from model import UNet
from dataset import ImageData
from tqdm import tqdm

# Hyperparameters
EPOCHS = 50
BATCH_SIZE = 1
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

image_dir = Path("train_images")
mask_dir = Path("train_masks")

dataset = ImageData(image_dir, mask_dir)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = UNet(in_channels=3, out_channels=1).to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

best_loss = float("inf")
best_model_path = CHECKPOINT_DIR / "best_model.pth"

for epoch in tqdm(range(1, EPOCHS + 1)):
    model.train()
    running_loss = 0.0
    
    for images, masks in train_loader:
        images, masks = images.to(DEVICE), masks.to(DEVICE)

        outputs = model(images)
        masks = torchvision.transforms.functional.center_crop(masks, outputs.shape[2:])
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch}/{EPOCHS}], Loss: {epoch_loss:.4f}")

    if epoch_loss < best_loss:
        best_loss = epoch_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"✅ Best model saved at epoch {epoch} with loss {best_loss:.4f}")

    if epoch % 10 == 0:
        ckpt_path = CHECKPOINT_DIR / f"model_epoch_{epoch}.pth"
        torch.save(model.state_dict(), ckpt_path)
        print(f"📦 Checkpoint saved at {ckpt_path}")

print("🎉 Training complete.")
