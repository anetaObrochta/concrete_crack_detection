import os
import random
import shutil


def count_files(directory):
    return len([f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))])


def create_sample_dataset(raw_dir, sample_dir, n_samples=5000):
    for class_name in ['positive', 'negative']:
        src_dir = os.path.join(raw_dir, class_name)
        dst_dir = os.path.join(sample_dir, class_name)

        print(f"\nProcessing {class_name} images:")
        print(f"Source directory: {src_dir}")
        print(f"Destination directory: {dst_dir}")

        # Ensure source directory exists
        if not os.path.exists(src_dir):
            print(f"Error: Source directory {src_dir} does not exist.")
            continue

        # Create destination directory if it doesn't exist
        os.makedirs(dst_dir, exist_ok=True)

        # Count and list all files
        all_files = [f for f in os.listdir(src_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
        print(f"Total {class_name} images in raw directory: {len(all_files)}")

        # Check existing files in sample directory
        existing_files = os.listdir(dst_dir)
        print(f"Existing files in sample {class_name} directory: {len(existing_files)}")

        if len(all_files) < n_samples:
            print(f"Warning: Only {len(all_files)} images found in {src_dir}. Using all available images.")
            sampled_files = all_files
        else:
            sampled_files = random.sample(all_files, n_samples)

        # Copy sampled files to sample directory
        copied_count = 0
        for file_name in sampled_files:
            src_path = os.path.join(src_dir, file_name)
            dst_path = os.path.join(dst_dir, file_name)
            if not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)
                copied_count += 1
            else:
                print(f"Warning: File {file_name} already exists in sample directory.")

        print(f"Newly copied {class_name} images: {copied_count}")

        # Final count in sample directory
        final_count = count_files(dst_dir)
        print(f"Final count in {class_name} sample directory: {final_count}")

    print("\nSample dataset creation complete")


if __name__ == "__main__":
    # Get the absolute path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Navigate up to the project root
    project_root = os.path.dirname(os.path.dirname(script_dir))

    # Define default paths relative to the project root
    default_raw_dir = os.path.join(project_root, "data", "raw")
    default_sample_dir = os.path.join(project_root, "data", "sample")

    # Allow user to input custom paths
    raw_dir = input(f"Enter path to raw data directory (default: {default_raw_dir}): ").strip() or default_raw_dir
    sample_dir = input(
        f"Enter path to sample data directory (default: {default_sample_dir}): ").strip() or default_sample_dir

    # Convert to absolute paths
    raw_dir = os.path.abspath(raw_dir)
    sample_dir = os.path.abspath(sample_dir)

    print(f"Using raw data from: {raw_dir}")
    print(f"Saving sample data to: {sample_dir}")

    create_sample_dataset(raw_dir, sample_dir)
