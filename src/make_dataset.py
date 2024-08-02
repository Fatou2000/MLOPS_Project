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


def show_random_images(images: np.ndarray, labels: np.ndarray, num_samples: int, mode: str = 'BGR2RGB'):
    if num_samples > len(images):
        raise ValueError("Number of samples requested exceeds available samples.")

    # Randomly select indices
    random_indices = random.sample(range(len(images)), num_samples)

    class_names={j:i for i,j in labels_.items()}

    selected_images = images[random_indices]
    selected_labels = [class_names[labels[index]].capitalize() for index in random_indices]

    # Define the number of rows and columns for displaying images
    num_cols = 3
    num_rows = (num_samples + num_cols - 1) // num_cols

    plt.figure(figsize=(15, 5 * num_rows))

    for i, (img, label) in enumerate(zip(selected_images, selected_labels)):
        plt.subplot(num_rows, num_cols, i + 1)
        if isinstance(mode,str) and not mode is None:
            mode=mode.upper()
        if mode == 'GRAY':
            img_display = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            plt.imshow(img_display, cmap='gray')
        elif mode == 'BGR2RGB':
            img_display = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img_display)
        elif mode == 'RGB':
            plt.imshow(img)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        plt.title(label)
        plt.axis('off')
    plt.tight_layout()
    plt.show()