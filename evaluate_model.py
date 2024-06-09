import matplotlib.pyplot as plt
import torch
import sys
from torchvision import transforms
from PIL import Image



from SwinIR.SwinIRmodels.network_swinir import SwinIR

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
    model = SwinIR(img_size=256, patch_size=1, in_chans=1, embed_dim=60, depths=[2, 2, 2, 2], num_heads=[2, 2, 2, 2],
                   window_size=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                   drop_path_rate=0.1, norm_layer=torch.nn.LayerNorm, ape=False, patch_norm=True, use_checkpoint=False,
                   upscale=1, img_range=1., upsampler='', resi_connection='1conv').to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
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
model_path = "best_model_experiment_1.pth"
input_image_path = r"test_images\barcode_76_col_8_aug_0.png"
ground_truth_image_path = r"test_images\barcode_76_col_8.png"

evaluate_model(model_path, input_image_path, ground_truth_image_path)
