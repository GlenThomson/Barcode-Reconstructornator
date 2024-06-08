import os
import random
import numpy as np
import cv2
from PIL import Image, ImageDraw
from pathlib import Path

def apply_random_noise(image):

    s_vs_p = 0.5
    amount = 0.010
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

def apply_random_rotation(image, max_angle=1):

    angle = random.uniform(-max_angle, max_angle)  # Rotate between -max_angle and max_angle degrees

    h, w = image.shape
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return rotated_image

def apply_random_shading(image):

    shaded_image = np.copy(image)
    white_shade = random.randint(230, 255)  # Narrowed range for white shade
    black_shade = random.randint(0, 25)  # Narrowed range for black shade

    shaded_image[shaded_image == 255] = white_shade
    shaded_image[shaded_image == 0] = black_shade
    return shaded_image


def apply_random_occlusion(image, max_occlusions=5):
    h, w = image.shape
    occluded_image = np.copy(image)
    for _ in range(random.randint(1, max_occlusions)):
        # Randomly determine occlusion color
        color = random.randint(0, 255)
        # Randomly determine the size and position of the occlusion
        occlusion_width = random.randint(10, 30)
        occlusion_height = random.randint(10, 30)
        x1 = random.randint(0, w - occlusion_width)
        y1 = random.randint(0, h - occlusion_height)
        x2 = x1 + occlusion_width
        y2 = y1 + occlusion_height
        # Draw the ellipse on the image
        cv2.ellipse(occluded_image, ((x1+x2)//2, (y1+y2)//2), (occlusion_width//2, occlusion_height//2), 0, 0, 360, color, -1)
    return occluded_image


def apply_random_curvature(image, max_curve=4):
    print("Applying random curvature")
    h, w = image.shape
    curve_strength = random.uniform(-max_curve, max_curve)
    print(f"Curve strength: {curve_strength}")
    src_points = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    dst_points = np.float32([[curve_strength, 0], [w - curve_strength, 0], [0, h], [w, h]])

    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    curved_image = cv2.warpPerspective(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return curved_image

def apply_random_transformations(image):
    transformations = [apply_random_noise, apply_random_blur, apply_random_shading,apply_random_occlusion]
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
