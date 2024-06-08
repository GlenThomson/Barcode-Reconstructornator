import os
import shutil
from barcode_generator import generate_dataset
from prepare_dataset import augment_and_prepare_datasets
from train_model import train_model

def remove_directory(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
        print(f"Removed directory: {directory}")

def main():
    run_name = "experiment_1"  # Specify a unique name for each run

    # Define paths
    generated_barcodes_dir = 'generated_barcodes'
    augmented_images_dir = 'augmented_images'
    train_input_dir = 'train_input'
    train_target_dir = 'train_target'
    val_input_dir = 'val_input'
    val_target_dir = 'val_target'
    
    # Remove directories if they exist
    remove_directory(generated_barcodes_dir)
    remove_directory(augmented_images_dir)
    remove_directory(train_input_dir)
    remove_directory(train_target_dir)
    remove_directory(val_input_dir)
    remove_directory(val_target_dir)
    
    # Generate barcodes
    print("Generating barcodes...")
    generate_dataset('generated_barcodes', num_samples=20000, columns=8, security_level=4)
    
    # Prepare datasets
    print("Preparing datasets...")
    augment_and_prepare_datasets('generated_barcodes', 'augmented_images', 'train_input', 'train_target', 'val_input', 'val_target', num_augmentations=1)
    
    # Train model
    print("Training model...")
    train_model(num_epochs=1,run_name=run_name)

if __name__ == "__main__":
    main()
