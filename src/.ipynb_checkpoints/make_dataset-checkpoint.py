import os
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter
import random

def load_data(path: str) -> tuple:
    images = []
    labels = []
    class_names = []
    labels_ = {}  # Dictionnaire pour mapper les noms de classe aux étiquettes numériques
    current_label = 0

    for class_name in os.listdir(path):
        class_path = os.path.join(path, class_name)
        if os.path.isdir(class_path):
            if class_name not in labels_:
                labels_[class_name] = current_label
                class_names.append(class_name)
                current_label += 1
            for img_name in tqdm(os.listdir(class_path), desc=f"Loading {class_name}"):
                img_path = os.path.join(class_path, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    images.append(img)
                    labels.append(labels_[class_name])
    images = np.array(images)
    labels = np.array(labels)
    return images, labels, class_names


def visualize_class_distribution(labels: np.ndarray, class_names: list):
    class_counts = Counter(labels)
    plt.figure(figsize=(10, 6))
    plt.bar(class_counts.keys(), class_counts.values(), tick_label=class_names)
    plt.xlabel('Classes')
    plt.ylabel('Counts')
    plt.title('Distribution des classes')
    plt.show()


def plot_images_from_subfolders(base_dir, num_images=3):
    subfolders = [os.path.join(base_dir, folder) for folder in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, folder))]

    for folder_path in subfolders:
        relative_path = os.path.relpath(folder_path, base_dir)
        print(f"Images de: {relative_path}")
        fig, axes = plt.subplots(nrows=1, ncols=num_images, figsize=(15, 5))
        files = os.listdir(folder_path)

        for i in range(num_images):
            img_path = os.path.join(folder_path, files[i])
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[i].imshow(img_rgb)
            axes[i].axis('off')
        plt.show()