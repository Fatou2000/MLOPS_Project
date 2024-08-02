import os
import numpy as np
import cv2
from tqdm import tqdm
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