import os
import random
import numpy as np
import cv2
from PIL import Image, ImageDraw
from pathlib import Path

def apply_random_noise(image):
    s_vs_p = 0.5
    amount = 0.004
    out = np.copy(image)
    # Salt mode
    num_salt = np.ceil(amount * image.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    out[tuple(coords)] = 255

    # Pepper mode
    num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    out[tuple(coords)] = 0

    return out

def apply_random_blur(image):
    ksize = random.choice([3, 5, 7])
    return cv2.GaussianBlur(image, (ksize, ksize), 0)

def apply_random_transformations(image):
    transformations = [apply_random_noise, apply_random_blur]
    random.shuffle(transformations)
    for transform in transformations:
        image = transform(image)
    return image

def augment_images(input_dir, output_dir, num_augmentations=10):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    image_paths = [os.path.join(input_dir, fname) for fname in os.listdir(input_dir) if fname.endswith('.png')]
    
    for image_path in image_paths:
        original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        for i in range(num_augmentations):
            augmented_image = apply_random_transformations(original_image.copy())
            output_path = os.path.join(output_dir, f"{base_name}_aug_{i}.png")
            cv2.imwrite(output_path, augmented_image)
            print(f"Saved {i+1} augmented images for {image_path}")

# Usage example
# augment_images('input_directory', 'output_directory', num_augmentations=10)
