import os
import shutil
import cv2
from pathlib import Path
from sklearn.model_selection import train_test_split
from PIL import Image
from data_augmentation import apply_random_transformations

def augment_and_prepare_datasets(clean_dir, aug_dir, train_input_dir, train_target_dir, val_input_dir, val_target_dir, val_split=0.2, num_augmentations=10):
    Path(train_input_dir).mkdir(parents=True, exist_ok=True)
    Path(train_target_dir).mkdir(parents=True, exist_ok=True)
    Path(val_input_dir).mkdir(parents=True, exist_ok=True)
    Path(val_target_dir).mkdir(parents=True, exist_ok=True)
    
    images = [os.path.join(clean_dir, fname) for fname in os.listdir(clean_dir) if fname.endswith('.png')]
    print(f"Total images found: {len(images)}")
    train_images, val_images = train_test_split(images, test_size=val_split, random_state=42)
    
    # Save clean images for validation targets and create noisy inputs
    print("Saving clean images for validation...")
    for image in val_images:
        img = Image.open(image).convert("L")
        img.save(os.path.join(val_target_dir, os.path.basename(image)))

        original_image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        noisy_image = apply_random_transformations(original_image.copy())
        noisy_image_path = os.path.join(val_input_dir, os.path.basename(image))
        cv2.imwrite(noisy_image_path, noisy_image)
        print(f"Saved noisy validation image: {noisy_image_path}")

    # Save clean images for training target
    print("Saving clean images for training target...")
    for image in train_images:
        img = Image.open(image).convert("L")
        img.save(os.path.join(train_target_dir, os.path.basename(image)))
    
    total_augmented_images = 0

    # Augment images for training input
    for image in train_images:
        print(f"Augmenting image: {image}")

        original_image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        base_name = os.path.splitext(os.path.basename(image))[0]
        
        # Create a temporary directory for the augmented images
        temp_aug_dir = os.path.join(aug_dir, 'temp_aug')
        Path(temp_aug_dir).mkdir(parents=True, exist_ok=True)

        for i in range(num_augmentations):
            augmented_image = apply_random_transformations(original_image.copy())
            aug_image_name = f"{base_name}_aug_{i}.png"
            aug_image_path = os.path.join(temp_aug_dir, aug_image_name)
            cv2.imwrite(aug_image_path, augmented_image)
            print(f"Saved augmented image: {aug_image_path}")

            # Move augmented images to train_input_dir
            new_aug_image_path = os.path.join(train_input_dir, aug_image_name)
            shutil.move(aug_image_path, new_aug_image_path)
            total_augmented_images += 1
            print(f"Moved augmented image: {new_aug_image_path}")

        # Remove the temporary directory and its contents
        shutil.rmtree(temp_aug_dir)
        print(f"Removed temporary directory: {temp_aug_dir}")

    print(f"Total augmented images generated: {total_augmented_images}")

# Example usage
# augment_and_prepare_datasets('generated_barcodes', 'augmented_images', 'train_input', 'train_target', 'val_input', 'val_target', num_augmentations=1)
