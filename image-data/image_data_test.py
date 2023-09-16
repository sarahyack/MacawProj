import os
import sys
import numpy as np
from PIL import Image
import pickle

sys.path.append("../")
from src.helper.config import folder4_path, folder5_path, folder6_path

def load_images(folder_path, label):
    """
    Load images from a specified folder path and assign a label to each image.

    Parameters:
        folder_path (str): The path to the folder containing the images.
        label (str): The label to assign to each image.

    Returns:
        tuple: A tuple containing two lists - `images` and `labels`.
            - `images` (list): A list of standardized image arrays.
            - `labels` (list): A list of labels corresponding to each image.
    """
    images = []
    labels = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            filepath = os.path.join(folder_path, filename)
            image = Image.open(filepath).convert("RGB").resize((28, 28))
            image_array = np.array(image)
            image_array_flatten = image_array.reshape(image_array.shape[0] * image_array.shape[1] * image_array.shape[2], 1)
            image_array_standardized = image_array_flatten / 255.0
            images.append(image_array_standardized)
            labels.append(label)
    return images, labels

# Folder paths and labels
folder1 = folder4_path
label1 = 0
folder2 = folder5_path
label2 = 1
folder3 = folder6_path
label3 = 2

# Load and label images from all three folders
images1, labels1 = load_images(folder1, label1)
images2, labels2 = load_images(folder2, label2)
images3, labels3 = load_images(folder3, label3)

# Combine images and labels from all folders
all_images = images1 + images2 + images3
all_labels = labels1 + labels2 + labels3

# Convert to NumPy arrays
all_images = np.array(all_images)
all_labels = np.array(all_labels)

# Save data to a file
with open("image_data_test.pkl", "wb") as f:
    pickle.dump((all_images, all_labels), f)

print("Data saved to image_data_test.pkl")
