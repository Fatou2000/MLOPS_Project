<<<<<<< HEAD
=======
"""Module for loading and preprocessing data"""

>>>>>>> origin/fatoumb
import os
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import Counter
import random
<<<<<<< HEAD
=======
from tensorflow.keras.preprocessing.image import img_to_array
import shutil
from tensorflow.keras.preprocessing.image import ImageDataGenerator

>>>>>>> origin/fatoumb

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


<<<<<<< HEAD
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
=======
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


def img_dimensions(base_dir):
     # Parcours de tous les sous-répertoires dans le répertoire de base
    for dir in os.listdir(base_dir):
        dir_path = os.path.join(base_dir, dir)
        if os.path.isdir(dir_path):
            print(f'Analyse des images en: {dir_path}')
            
            # Liste des fichiers dans le sous-répertoire courant
            files = os.listdir(dir_path)
            dim = []
            
            for file in files:
                img_path = os.path.join(dir_path, file)
                img = cv2.imread(img_path)
                
                # Si l'image est bien lue, on extrait ses dimensions
                if img is not None:
                    height, width, channels = img.shape
                    dim.append((height, width))
            
            # Comptage des dimensions les plus courantes
            count_dim = Counter(dim)
            print("Dimensions les plus courantes:")
            for dim, freq in count_dim.most_common(15):
                print(f"Dimension (hauteur x largeur): {dim}, Fréquence: {freq}")
            print('\n')


def process_dataset(source_dir, processed_dir):
    # Créer le répertoire de destination s'il n'existe pas
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    data = []
    labels = []

    # Fixe la graine aléatoire pour la reproductibilité
    random.seed(42)

    # Parcourir chaque classe dans le répertoire source
    for class_name in sorted(os.listdir(source_dir)):
        class_dir = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        # Lister tous les fichiers de la classe
        files = [f for f in os.listdir(class_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

        # Mélange aléatoirement l'ordre des fichiers
        random.shuffle(files)

        # Fonction pour traiter les images et enregistrer les images redimensionnées
        def process_and_save_files(file_list, dest_dir):
            class_dest_dir = os.path.join(dest_dir, class_name)
            if not os.path.exists(class_dest_dir):
                os.makedirs(class_dest_dir)
            for file in file_list:
                try:
                    # Charger l'image avec OpenCV
                    image_path = os.path.join(class_dir, file)
                    image = cv2.imread(image_path)
                    # Redimensionner l'image à 128x128 pixels
                    image_resized = cv2.resize(image, (128, 128))
                    # Convertir l'image en tableau numpy
                    image_array = img_to_array(image_resized)
                    # Ajouter l'image traitée à la liste
                    data.append(image_array)
                    # Ajouter l'étiquette (nom du dossier) à la liste
                    labels.append(class_name)
                    
                    # Chemin de l'image redimensionnée à enregistrer
                    save_path = os.path.join(class_dest_dir, file)
                    # Enregistrer l'image redimensionnée
                    cv2.imwrite(save_path, image_resized)
                except Exception as e:
                    # print(f"Error processing image {file} in category {class_name}: {str(e)}")
                    continue

        # Traiter les fichiers et enregistrer les images redimensionnées dans le répertoire approprié
        process_and_save_files(files, processed_dir)

        print(f"Classe {class_name} : {len(files)} processed")

    # Convertir les listes data et labels en tableaux numpy
    data = np.array(data, dtype="float32")
    labels = np.array(labels)

    return data, labels


def increase_dataset(data, labels, zoom_range=0.2, horizontal_flip=True, augmentation_ratio=1/3):
    # Créer un générateur d'augmentation d'images
    datagen = ImageDataGenerator(zoom_range=zoom_range, horizontal_flip=horizontal_flip)

    # Initialiser les listes pour stocker les images et les labels augmentés
    augmented_images = []
    augmented_labels = []

    # Calculer le nombre d'images à générer
    augmentation_count = int(len(data) * augmentation_ratio)

    # Boucle pour chaque image et label
    for img, label in zip(data, labels):
        # Préparer l'image pour l'augmentation
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        
        # Générer des images augmentées
        i = 0
        for batch in datagen.flow(img, batch_size=1):
            augmented_images.append(batch[0].astype('uint8'))
            augmented_labels.append(label)
            i += 1
            if i >= (augmentation_count / len(data)):
                break

    # Convertir les listes en tableaux numpy
    augmented_images = np.array(augmented_images)
    augmented_labels = np.array(augmented_labels)

    # Combiner les données originales avec les données augmentées
    final_imgs_data = np.concatenate((data, augmented_images), axis=0)
    final_labels_data = np.concatenate((labels, augmented_labels), axis=0)

    return final_imgs_data, final_labels_data
>>>>>>> origin/fatoumb
