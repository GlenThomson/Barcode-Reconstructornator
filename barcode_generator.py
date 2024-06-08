import os
import random
import pdf417gen
from PIL import Image
from pathlib import Path

def generate_pdf417_barcode(data, output_path, columns, security_level=5):
    codes = pdf417gen.encode(data, columns=columns, security_level=security_level)
    image = pdf417gen.render_image(codes, scale=3, ratio=3, padding=5)
    
    # Rotate the image to be vertical
    image = image.rotate(90, expand=True)
    image.save(output_path)

def generate_dataset(output_dir, num_samples=1000, columns=8, security_level=5):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for i in range(num_samples):
        data = f"Sample data {i}"
        barcode_path = os.path.join(output_dir, f"barcode_{i}_col_{columns}.png")

        generate_pdf417_barcode(data, barcode_path, columns, security_level)

# Generate the dataset with a fixed column value
generate_dataset('generated_barcodes', num_samples=2000, columns=8, security_level=4)
