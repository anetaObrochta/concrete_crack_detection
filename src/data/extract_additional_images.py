import os
import random
import shutil


# Extract from the raw dataset more images that are different than the sample data used for
# model training, validation, and testing

def extract_additional_images(raw_dir, sample_dir, output_dir, n_images=100):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'positive'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'negative'), exist_ok=True)

    sample_images = set()
    for root, _, files in os.walk(sample_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                sample_images.add(file)

    raw_images = {'positive': [], 'negative': []}
    for category in ['positive', 'negative']:
        category_dir = os.path.join(raw_dir, category)
        for root, _, files in os.walk(category_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')) and file not in sample_images:
                    raw_images[category].append(os.path.join(root, file))

    for category in ['positive', 'negative']:
        n_category_images = min(n_images // 2, len(raw_images[category]))
        selected_images = random.sample(raw_images[category], n_category_images)
        for img_path in selected_images:
            dst_path = os.path.join(output_dir, category, os.path.basename(img_path))
            shutil.copy2(img_path, dst_path)

    print(f"Extracted {len(os.listdir(os.path.join(output_dir, 'positive')))} positive images")
    print(f"Extracted {len(os.listdir(os.path.join(output_dir, 'negative')))} negative images")


# Usage
if __name__ == "__main__":
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate up to the project root
    project_root = os.path.dirname(os.path.dirname(script_dir))
    # Define default paths relative to the project root
    default_raw_dir = os.path.join(project_root, "data", "raw")
    default_sample_dir = os.path.join(project_root, "data", "sample")
    default_output_dir = os.path.join(project_root, "data", "additional_images")

    # Allow user to input custom paths
    raw_dir = input(f"Enter path to raw data directory (default: {default_raw_dir}): ").strip()
    sample_dir = input(f"Enter path to sample data directory (default: {default_sample_dir}): ").strip()
    output_dir = input(f"Enter path to output data directory (default: {default_output_dir}): ").strip()

    # Use default paths if no input is provided
    raw_dir = raw_dir or default_raw_dir
    sample_dir = sample_dir or default_sample_dir
    output_dir = output_dir or default_output_dir

    # Remove any quotation marks from the input paths
    raw_dir = raw_dir.strip('"').strip("'")
    sample_dir = sample_dir.strip('"').strip("'")
    output_dir = output_dir.strip('"').strip("'")

    # Convert to absolute paths
    raw_dir = os.path.abspath(raw_dir)
    sample_dir = os.path.abspath(sample_dir)
    output_dir = os.path.abspath(output_dir)

    print(f"Using raw data from: {raw_dir}")
    print(f"Using sample data from: {sample_dir}")
    print(f"Saving additional images to: {output_dir}")

    extract_additional_images(raw_dir, sample_dir, output_dir)