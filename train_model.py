import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import json
import matplotlib.pyplot as plt

from SwinIR.SwinIRmodels.network_swinir import SwinIR

class BarcodeDataset(Dataset):
    def __init__(self, input_dir, target_dir, transform=None):
        self.input_dir = input_dir
        self.target_dir = target_dir
        self.transform = transform
        self.input_images = sorted(os.listdir(input_dir))
        self.target_images = sorted(os.listdir(target_dir))

        assert len(self.input_images) == len(self.target_images), "Mismatch between input and target images"

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        input_image_path = os.path.join(self.input_dir, self.input_images[idx])
        target_image_path = os.path.join(self.target_dir, self.target_images[idx])
        
        input_image = Image.open(input_image_path).convert("L")
        target_image = Image.open(target_image_path).convert("L")
        
        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)
        
        return input_image, target_image

def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def train_model(num_epochs=2, batch_size=2, learning_rate=0.001, run_name="experiment"):
    train_input_dir = 'train_input'
    train_target_dir = 'train_target'
    val_input_dir = 'val_input'
    val_target_dir = 'val_target'

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = BarcodeDataset(train_input_dir, train_target_dir, transform=transform)
    val_dataset = BarcodeDataset(val_input_dir, val_target_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SwinIR(img_size=256, patch_size=1, in_chans=1, embed_dim=60, depths=[2, 2, 2, 2], num_heads=[2, 2, 2, 2],
                   window_size=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                   drop_path_rate=0.1, norm_layer=torch.nn.LayerNorm, ape=False, patch_norm=True, use_checkpoint=False,
                   upscale=1, img_range=1., upsampler='', resi_connection='1conv').to(device)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:  # Print every 10 batches
                print(f"[Epoch {epoch + 1}, Batch {i + 1}] Training Loss: {running_loss / 10:.4f}")
                running_loss = 0.0

        train_losses.append(running_loss / len(train_loader))

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        print(f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"best_model_{run_name}.pth")

    torch.save(model.state_dict(), f"final_model_{run_name}.pth")
    print("Finished Training")

    plot_losses(train_losses, val_losses)