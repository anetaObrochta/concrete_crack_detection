import os
import random
import shutil

def split_data(sample_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    # Ensure ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1"

    # Create output directories
    for split in ['train', 'validation', 'test']:
        for class_name in ['positive', 'negative']:
            os.makedirs(os.path.join(output_dir, split, class_name), exist_ok=True)

    for class_name in ['positive', 'negative']:
        # Get all files
        class_dir = os.path.join(sample_dir, class_name)
        all_files = os.listdir(class_dir)
        random.shuffle(all_files)

        # Calculate split sizes
        n_files = len(all_files)
        n_train = int(n_files * train_ratio)
        n_val = int(n_files * val_ratio)

        # Split files
        train_files = all_files[:n_train]
        val_files = all_files[n_train:n_train+n_val]
        test_files = all_files[n_train+n_val:]

        # Copy files to respective directories
        for split, files in [('train', train_files), ('validation', val_files), ('test', test_files)]:
            for file in files:
                src = os.path.join(class_dir, file)
                dst = os.path.join(output_dir, split, class_name, file)
                shutil.copy2(src, dst)

        print(f"{class_name.capitalize()} data split:")
        print(f"  Train: {len(train_files)}")
        print(f"  Validation: {len(val_files)}")
        print(f"  Test: {len(test_files)}")

if __name__ == "__main__":
    # Define paths
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sample_dir = os.path.join(project_root, "data", "sample")
    output_dir = os.path.join(project_root, "data", "processed")

    # Get user confirmation
    print(f"This script will split data from {sample_dir}")
    print(f"and create train/val/test splits in {output_dir}")
    confirm = input("Do you want to proceed? (y/n): ").lower().strip()

    if confirm == 'y':
        split_data(sample_dir, output_dir)
        print("Data split complete!")
    else:
        print("Operation cancelled.")
