import os
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from unet_model import UNet

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
    

def train_model(run_name, num_epochs=2, batch_size=16, learning_rate=0.001, checkpoint_path=None):
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
    model = UNet(in_channels=1, out_channels=1).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    start_epoch = 0
    best_val_loss = float('inf')
    training_results = {
        'run_name': run_name,
        'epoch_losses': [],
        'val_losses': []
    }

    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        if 'model_state_dict' in checkpoint and 'optimizer_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['best_val_loss']
            print(f"Loaded checkpoint from {checkpoint_path}, starting at epoch {start_epoch}")
        else:
            print(f"Checkpoint file {checkpoint_path} is missing required keys, starting training from scratch.")
    else:
        print(f"No checkpoint found at {checkpoint_path}, starting training from scratch.")

    for epoch in range(start_epoch, start_epoch + num_epochs):
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
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 10:.3f}")
                running_loss = 0.0

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
        print(f"Epoch {epoch + 1}, Validation Loss: {val_loss:.3f}")

        training_results['epoch_losses'].append(loss.item())
        training_results['val_losses'].append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss
            }, f"best_barcode_reconstruction_model_{run_name}.pth")

    # Ensure the last epoch variable is defined for final checkpoint saving
    final_epoch = epoch if 'epoch' in locals() else start_epoch + num_epochs - 1
    torch.save({
        'epoch': final_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss
    }, f"final_barcode_reconstruction_model_{run_name}.pth")
    print("Finished Training")

    # Save training results to a file
    results_file = f"{run_name}_results.json"
    with open(results_file, 'w') as f:
        json.dump(training_results, f, indent=4)
    print(f"Saved training results to {results_file}")
