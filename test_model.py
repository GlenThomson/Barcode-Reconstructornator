import os
import torch
from PIL import Image
import torchvision.transforms as transforms
from unet_model import UNet  # Ensure this matches your model file name and class name

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path):
    model = UNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def process_image(image_path, model, transform):
    image = Image.open(image_path).convert("L")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
    output = output.squeeze().cpu().numpy()
    return output

def save_image(image_array, output_path):
    image = Image.fromarray((image_array * 255).astype("uint8"))
    image.save(output_path)

def test_model(model_path, test_dir, output_dir):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    model = load_model(model_path)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    test_images = [os.path.join(test_dir, fname) for fname in os.listdir(test_dir) if fname.endswith('.png')]
    
    for image_path in test_images:
        output_image = process_image(image_path, model, transform)
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        save_image(output_image, output_path)
        print(f"Processed {image_path} and saved to {output_path}")

# Example usage
test_model("best_barcode_reconstruction_model.pth", "test_images", "output_images")
