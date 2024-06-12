import os
import random
import pdf417gen
from PIL import Image, ImageOps
from pathlib import Path
import cv2
import numpy as np

def apply_random_rotation(image, max_angle=2):
    angle = random.uniform(-max_angle, max_angle)
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return rotated_image

def apply_random_curvature(image, max_curve=4):
    h, w = image.shape[:2]
    curve_strength = random.uniform(-max_curve, max_curve)
    src_points = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    dst_points = np.float32([[curve_strength, 0], [w - curve_strength, 0], [0, h], [w, h]])
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    curved_image = cv2.warpPerspective(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return curved_image

def apply_random_shift(image, max_shift=10):
    h, w = image.shape[:2]
    shift_x = random.randint(-max_shift, max_shift)
    shift_y = random.randint(-max_shift, max_shift)
    matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted_image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return shifted_image

def apply_transformations(image):
    image = apply_random_rotation(image)
    image = apply_random_curvature(image)
    image = apply_random_shift(image)
    return image

def generate_pdf417_barcode(data, output_path, columns, security_level=5, scale=3, padding=20):
    codes = pdf417gen.encode(data, columns=columns, security_level=security_level)
    image = pdf417gen.render_image(codes, scale=scale, ratio=3, padding=padding + 100)  # Add extra padding
    
    # Rotate the image to be vertical
    image = image.rotate(90, expand=True)
    
    # Convert PIL image to numpy array
    image = np.array(image)
    
    # Apply transformations
    image = apply_transformations(image)
    
    # Convert back to PIL image
    image = Image.fromarray(image)
    
    # Crop the image back to the desired padding
    width, height = image.size
    image = image.crop((100, 100, width - 85, height - 85))
    
    # Resize the image to the desired dimensions (180 width by 1000 height)
    image = image.resize((180, 1000), Image.LANCZOS)
    
    # Ensure the background is white
    image = np.array(image)
    image[image == 255] = 255  # Set all white pixels to pure white
    
    # Convert back to PIL image and save
    image = Image.fromarray(image)
    image.save(output_path)

def generate_dataset(output_dir, num_samples=10, column_range=(5, 6), security_range=(3, 4), scale_range=(5, 7), padding_range=(20, 30)):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for i in range(num_samples):
        data = f"Sample data {i}"
        columns = random.randint(*column_range)
        security_level = random.randint(*security_range)
        scale = random.randint(*scale_range)
        padding = random.randint(*padding_range)
        barcode_path = os.path.join(output_dir, f"barcode_{i}_col_{columns}_sec_{security_level}_scale_{scale}_pad_{padding}.png")
        
        generate_pdf417_barcode(data, barcode_path, columns, security_level, scale, padding)

# Example usage
# generate_dataset('generated_barcodes', num_samples=10, column_range=(5, 7), security_range=(3, 4), scale_range=(5, 7), padding_range=(20, 30))
