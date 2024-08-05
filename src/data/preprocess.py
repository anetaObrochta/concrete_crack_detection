import numpy as np
from PIL import Image
import os
import argparse
from tensorflow import keras

def load_and_preprocess_image(image_path):
    """Load an image and preprocess it"""
    image = Image.open(image_path)
    image = np.array(image) / 255.0  # Normalize to [0, 1]
    return image

def create_data_generator(rotation_range=20, horizontal_flip=True, width_shift_range=0.2, height_shift_range=0.2):
    """Create a data generator for augmentation"""
    return keras.preprocessing.image.ImageDataGenerator(
        rotation_range=rotation_range,
        horizontal_flip=horizontal_flip,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        preprocessing_function=lambda x: x / 255.0  # Normalize to [0, 1]
    )

def preprocess_dataset(input_dir, output_dir):
    """Preprocess all images in a directory and save them"""
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            image = load_and_preprocess_image(input_path)
            Image.fromarray((image * 255).astype(np.uint8)).save(output_path)

    print(f"Preprocessing complete. Processed images saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess images for concrete crack detection.")
    parser.add_argument("input_dir", help="Path to the input directory containing images")
    parser.add_argument("output_dir", help="Path to the output directory for processed images")
    args = parser.parse_args()

    preprocess_dataset(args.input_dir, args.output_dir)