"""
This file contains the code for loading 
and preprocessing the Stanford Cars dataset.
or any other custom dataset.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import StanfordCars, ImageFolder
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import shutil
import os


def load_stanford_cars_dataset(
    root: str = './data',
    download: bool = True,
    batch_size: int = 4,
    img_size: int = 64
) -> tuple[DataLoader, DataLoader]:
    """
    Downloads and loads the Stanford Cars dataset using torchvision.

    Args:
        root (str): Root directory for the dataset.
        download (bool): If True, downloads the dataset if not present.
        batch_size (int): Batch size for the DataLoader.
        img_size (int): The desired size for the images (resized to img_size x img_size).

    Returns:
        tuple[DataLoader, DataLoader]: A tuple containing the training and testing DataLoaders.
    """
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    try:
        train_dataset = torchvision.datasets.StanfordCars(
            root=root,
            split='train',
            download=download,
            transform=transform
        )
        test_dataset = torchvision.datasets.StanfordCars(
            root=root,
            split='test',
            download=download,
            transform=transform
        )

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, test_loader

    except Exception as e:
        print(f"Could not load dataset: {e}")
        print("Please ensure the dataset is available or download is permitted.")
        return None, None


def load_custom_image_folder(
    train_dir: str,
    test_dir: str,
    img_size: int = 64,
    batch_size: int = 4
    ) -> tuple[DataLoader | None, DataLoader | None]:
    """
    Loads images from a custom directory structure using torchvision.datasets.ImageFolder.
    Expects pre-defined 'train' and 'test' directories with class subfolders,
    as described in README.md.
    Example:
        train_dir/class_a/image1.jpg
        train_dir/class_b/image2.png
        test_dir/class_a/image3.jpg

    Args:
        train_dir (str): Path to the training directory (containing class subfolders).
        test_dir (str): Path to the testing directory (containing class subfolders).
        img_size (int): The desired size for the images (resized to img_size x img_size).
        batch_size (int): Batch size for the DataLoader.

    Returns:
        tuple[DataLoader | None, DataLoader | None]: A tuple containing train and test DataLoaders,
                                                   or (None, None) if directories are invalid.
    """

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    train_loader = None
    test_loader = None

    try:
        if not os.path.isdir(train_dir):
            print(f"Error: Training directory not found or is not a directory: {train_dir}")
            return None, None
        
        train_dataset = ImageFolder(root=train_dir, transform=transform)
        if len(train_dataset) == 0:
             print(f"Warning: No images found in {train_dir} or its subdirectories.")
             return None, None
        if len(train_dataset.classes) == 0:
             print(f"Warning: No class subdirectories found in {train_dir}. ImageFolder expects root/class/image.jpg structure.")
             # If no class folders, ImageFolder might still load images directly in root, 
             # but it's not the intended use and lacks labels.
             # Consider adding an error here if class folders are strictly required.

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        print(f"Loaded {len(train_dataset)} images from {len(train_dataset.classes)} classes in {train_dir}")

    except Exception as e:
        print(f"Error loading training data from {train_dir}: {e}")
        return None, None # Return None for both if train loading fails

    # Load test data if test_dir is provided and valid
    try:
        if test_dir and os.path.isdir(test_dir):
            test_dataset = ImageFolder(root=test_dir, transform=transform)
            if len(test_dataset) > 0:
                if len(test_dataset.classes) == 0:
                     print(f"Warning: No class subdirectories found in {test_dir}. Using root as single class.")
                test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
                print(f"Loaded {len(test_dataset)} images from {len(test_dataset.classes)} classes in {test_dir}")
            else:
                print(f"Warning: No images found in {test_dir} or its subdirectories. Test loader will be None.")
        elif test_dir:
            print(f"Warning: Test directory not found or is not a directory: {test_dir}. Test loader will be None.")
        else:
             print("No test directory provided. Test loader will be None.")

    except Exception as e:
        print(f"Error loading testing data from {test_dir}: {e}")
        # Don't necessarily return None for train_loader if test fails

    return train_loader, test_loader

if __name__ == "__main__":
    # Example usage assuming data is in ./custom/train and ./custom/test
    # Create dummy structure for testing if needed:
    # os.makedirs("./custom/train/class1", exist_ok=True)
    # os.makedirs("./custom/test/class1", exist_ok=True)
    # with open("./custom/train/class1/dummy1.txt", "w") as f: f.write("dummy") # ImageFolder ignores non-image files
    # with open("./custom/test/class1/dummy2.txt", "w") as f: f.write("dummy")

    print("Testing load_custom_image_folder...")
    # train_loader = load_custom_image_folder( # Old call
    #     source_dir='./custom',
    #     train_dir='./custom/train',
    #     test_dir='./custom/test',
    # )
    train_loader, test_loader = load_custom_image_folder(
        train_dir='./custom/train', # Expects ./custom/train/class_a/...
        test_dir='./custom/test',   # Expects ./custom/test/class_a/...
        img_size=64,
        batch_size=4
    )

    if train_loader:
        print(f"Train loader created. Number of batches: {len(train_loader)}")
        # Example: Iterate over one batch
        # try:
        #     images, labels = next(iter(train_loader))
        #     print("Sample batch - Images shape:", images.shape, "Labels:", labels)
        # except StopIteration:
        #     print("Train loader is empty.")

    if test_loader:
        print(f"Test loader created. Number of batches: {len(test_loader)}")
