import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.utils import load_img, img_to_array


def get_project_root():
    """Returns project root folder."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_data_dir():
    """Returns the path to the data directory."""
    return os.path.join(get_project_root(), 'data', 'sample')


# Adds the project root to the Python path
project_root = get_project_root()
sys.path.append(project_root)

# Use this function to get the data directory path
data_dir = get_data_dir()
print(f"Data directory: {data_dir}")


def load_data(data_dir):
    images = []
    labels = []

    for class_name in ['positive', 'negative']:
        class_dir = os.path.join(data_dir, class_name)
        label = 1 if class_name == 'positive' else 0
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img = load_img(img_path, target_size=(227, 227))  # Resize to original dimensions
            img_array = img_to_array(img)
            images.append(img_array)
            labels.append(label)

    return np.array(images), np.array(labels)


def prepare_data(data_dir, test_size=0.2, val_size=0.2):
    X, y = load_data(data_dir)

    # First split: separate test set
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=42)

    # Second split: separate validation set from training set
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=val_size,
                                                      stratify=y_train_val, random_state=42)

    # Normalize the data
    X_train = X_train / 255.0
    X_val = X_val / 255.0
    X_test = X_test / 255.0

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(data_dir)

    print(f"Training set shape: {X_train.shape}")
    print(f"Validation set shape: {X_val.shape}")
    print(f"Test set shape: {X_test.shape}")

    # Save the prepared data in the data directory
    save_dir = os.path.join(get_project_root(), 'data', 'processed')
    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(save_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(save_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(save_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(save_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(save_dir, 'y_test.npy'), y_test)

    print(f"Processed data saved in: {save_dir}")

    # Verify TensorFlow GPU usage
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    print("TensorFlow version:", tf.__version__)