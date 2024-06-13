import os
import random
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

def apply_random_noise(image):
    s_vs_p = 0.5
    amount = 0.030
    out = np.copy(image)
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    out[tuple(coords)] = 255
    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    out[tuple(coords)] = 0
    return out

def apply_random_blur(image):
    ksize = random.choice([3, 5, 7])
    return cv2.GaussianBlur(image, (ksize, ksize), 0)

def apply_random_shading(image):
    shaded_image = np.copy(image)
    white_shade = random.randint(230, 255)
    black_shade = random.randint(0, 25)
    shaded_image[shaded_image == 255] = white_shade
    shaded_image[shaded_image == 0] = black_shade
    return shaded_image

def apply_random_glare(image, max_occlusions=5):
    h, w = image.shape
    glared_image = np.copy(image)
    for _ in range(random.randint(1, max_occlusions)):
        glare_width = random.randint(10, 30)
        glare_height = random.randint(10, 300)
        x1 = random.randint(0, w - glare_width)
        y1 = random.randint(0, h - glare_height)
        overlay = glared_image.copy()
        alpha = random.uniform(0.2, 0.6)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask, ((x1 + glare_width // 2), (y1 + glare_height // 2)), (glare_width // 2, glare_height // 2), 0, 0, 360, 255, -1)
        angle = random.uniform(0, 360)
        matrix = cv2.getRotationMatrix2D(((x1 + glare_width // 2), (y1 + glare_height // 2)), angle, 1)
        rotated_mask = cv2.warpAffine(mask, matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        overlay[rotated_mask == 255] = 255
        glared_image = cv2.addWeighted(overlay, alpha, glared_image, 1 - alpha, 0)
    return glared_image

def apply_random_occlusion(image, max_occlusions=5):
    h, w = image.shape
    occluded_image = np.copy(image)
    for _ in range(random.randint(1, max_occlusions)):
        color = random.randint(0, 255)
        occlusion_width = random.randint(10, 20)
        occlusion_height = random.randint(10, 20)
        x1 = random.randint(0, w - occlusion_width)
        y1 = random.randint(0, h - occlusion_height)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.ellipse(mask, ((x1 + occlusion_width // 2), (y1 + occlusion_height // 2)), (occlusion_width // 2, occlusion_height // 2), 0, 0, 360, 255, -1)
        angle = random.uniform(0, 360)
        matrix = cv2.getRotationMatrix2D(((x1 + occlusion_width // 2), (y1 + occlusion_height // 2)), angle, 1)
        rotated_mask = cv2.warpAffine(mask, matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        occluded_image[rotated_mask == 255] = color
    return occluded_image

def add_random_text(image, num_texts=1):
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    width, height = pil_image.size
    font_size = random.randint(30, 45)
    font = ImageFont.truetype("arial.ttf", font_size)
    
    for _ in range(num_texts):
        text = ''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789', k=random.randint(5, 10)))
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
        
        if text_width < width and text_height < height:
            x = random.randint(0, width - text_width)
            y = random.randint(0, height - text_height)
            angle = random.uniform(-45, 45)

            text_image = Image.new('RGBA', (text_width, text_height), (255, 255, 255, 0))
            text_draw = ImageDraw.Draw(text_image)
            text_draw.text((0, 0), text, fill=(0, 0, 0, 255), font=font)
            text_image = text_image.rotate(angle, expand=True)

            pil_image.paste(text_image, (x, y), text_image)
    
    return np.array(pil_image)

def change_background_color(image, intensity_range=(100, 200)):
    background_intensity = random.randint(*intensity_range)
    image[image == 255] = background_intensity
    return image

def apply_random_transparency(image, max_transparency=0.5):
    alpha = random.uniform(0, max_transparency)
    transparent_image = image * (1 - alpha) + 255 * alpha
    return transparent_image.astype(np.uint8)

def apply_random_transformations(image):
    # Apply background color change first
    image = change_background_color(image)
    transformations = [apply_random_noise, apply_random_blur, apply_random_shading, apply_random_occlusion, add_random_text, apply_random_glare, apply_random_transparency]
    random.shuffle(transformations)
    for transform in transformations:
        image = transform(image)
    return image

def augment_images(input_dir, output_dir, num_augmentations=10):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    image_paths = [os.path.join(input_dir, fname) for fname in os.listdir(input_dir) if fname.endswith('.png')]
    
    for image_path in image_paths:
        print(f"Processing image: {image_path}")
        original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        for i in range(num_augmentations):
            print(f"Augmenting image {i+1}/{num_augmentations} for {image_path}")
            augmented_image = apply_random_transformations(original_image.copy())
            output_path = os.path.join(output_dir, f"{base_name}_aug_{i}.png")
            cv2.imwrite(output_path, augmented_image)
            print(f"Saved {i+1} augmented image for {image_path} to {output_path}")

# Usage example
# augment_images('input_directory', 'output_directory', num_augmentations=10)
