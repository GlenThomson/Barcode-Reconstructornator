import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from unet_model import UNet
from PIL import Image

def load_image(image_path, transform=None):
    image = Image.open(image_path).convert("L")
    if transform:
        image = transform(image)
    return image

def show_images(input_image, ground_truth_image, predicted_image):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(input_image.squeeze(), cmap='gray')
    axes[0].set_title("Input Image")
    axes[1].imshow(ground_truth_image.squeeze(), cmap='gray')
    axes[1].set_title("Ground Truth Image")
    axes[2].imshow(predicted_image.squeeze(), cmap='gray')
    axes[2].set_title("Predicted Image")
    plt.show()

def evaluate_model(model_path, input_image_path, ground_truth_image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    transform = transforms.Compose([transforms.ToTensor()])
    input_image = load_image(input_image_path, transform).unsqueeze(0).to(device)
    ground_truth_image = load_image(ground_truth_image_path, transform).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_image)
    
    input_image = input_image.cpu().numpy().transpose(0, 2, 3, 1)
    ground_truth_image = ground_truth_image.cpu().numpy().transpose(0, 2, 3, 1)
    output_image = output.cpu().numpy().transpose(0, 2, 3, 1)
    
    show_images(input_image, ground_truth_image, output_image)

# Path to the model and test images
model_path = "best_barcode_reconstruction_model.pth"
input_image_path = r"val_input\barcode_17_col_8.png"
ground_truth_image_path = r"val_target\barcode_17_col_8.png"

evaluate_model(model_path, input_image_path, ground_truth_image_path)

