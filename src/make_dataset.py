"""Module for loading and preprocessing data"""

import os
import random
from collections import Counter
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tqdm import tqdm


def load_data(path: str) -> tuple:
    """
    Charge les images et les étiquettes à partir d'un répertoire structuré en 
    sous-dossiers représentant différentes classes.

    Arguments:
    path -- Chemin du répertoire contenant les sous-dossiers d'images.

    Retourne:
    images -- Numpy array contenant toutes les images chargées.
    labels -- Numpy array contenant les étiquettes correspondantes des images.
    class_names -- Liste des noms de classes correspondant aux étiquettes 
    numériques.
    """
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
    """
    Visualise la distribution des classes dans le dataset en créant un 
    histogramme.

    Arguments:
    labels -- Numpy array contenant les étiquettes des images.
    class_names -- Liste des noms des classes correspondant aux étiquettes.

    Retourne:
    Rien. Affiche un graphique de la distribution des classes.
    """
    class_counts = Counter(labels)
    plt.figure(figsize=(10, 6))
    plt.bar(class_counts.keys(), class_counts.values(), tick_label=class_names)
    plt.xlabel('Classes')
    plt.ylabel('Counts')
    plt.title('Distribution des classes')
    plt.show()


def plot_images_from_subfolders(base_dir, num_images=3):
    """
    Affiche un échantillon d'images à partir de chaque sous-dossier dans un 
    répertoire de base spécifié.

    Arguments:
    base_dir -- Chemin du répertoire de base contenant les sous-dossiers 
    d'images.
    num_images -- Nombre d'images à afficher par sous-dossier (par défaut 3).

    Retourne:
    Rien. Affiche les images des sous-dossiers.
    """
    subfolders = [
        os.path.join(base_dir, folder) 
        for folder in os.listdir(base_dir) 
        if os.path.isdir(os.path.join(base_dir, folder))
    ]

    for folder_path in subfolders:
        relative_path = os.path.relpath(folder_path, base_dir)
        print(f"Images de: {relative_path}")
        _, axes = plt.subplots(nrows=1, ncols=num_images, figsize=(15, 5))
        files = os.listdir(folder_path)

        for i in range(num_images):
            img_path = os.path.join(folder_path, files[i])
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[i].imshow(img_rgb)
            axes[i].axis('off')
        plt.show()


def img_dimensions(base_dir):
    """
    Analyse les dimensions des images dans les sous-répertoires d'un répertoire
     de base.

    Arguments:
    base_dir -- Chemin du répertoire de base contenant les sous-dossiers 
    d'images.

    Retourne:
    Rien. Affiche les dimensions les plus courantes des images.
    """
    for dir_name in os.listdir(base_dir):
        dir_path = os.path.join(base_dir, dir_name)
        if os.path.isdir(dir_path):
            print(f'Analyse des images en: {dir_path}')
            
            files = os.listdir(dir_path)
            dim = []
            
            for file_name in files:
                img_path = os.path.join(dir_path, file_name)
                img = cv2.imread(img_path)
                
                if img is not None:
                    height, width, _ = img.shape
                    dim.append((height, width))
            
            count_dim = Counter(dim)
            print("Dimensions les plus courantes:")
            for dimension, freq in count_dim.most_common(15):
                print(f"Dimension (hauteur x largeur): {dimension}, Fréquence: {freq}")
            print('\n')


def process_dataset(source_dir, processed_dir):
    """
    Traite un dataset d'images en les redimensionnant à une taille fixe de 
    128x128 pixels et les enregistre dans un nouveau répertoire.

    Arguments:
    source_dir -- Chemin du répertoire source contenant les images à traiter.
    processed_dir -- Chemin du répertoire où les images traitées seront 
    enregistrées.

    Retourne:
    data -- Numpy array contenant les images traitées.
    labels -- Numpy array contenant les étiquettes correspondantes des images.
    """
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    data = []
    labels = []

    random.seed(42)

    for class_name in sorted(os.listdir(source_dir)):
        class_dir = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        files = [f for f in os.listdir(class_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

        def process_and_save_files(file_list, dest_dir):
            class_dest_dir = os.path.join(dest_dir, class_name)
            if not os.path.exists(class_dest_dir):
                os.makedirs(class_dest_dir)
            for file in file_list:
                try:
                    image_path = os.path.join(class_dir, file)
                    image = cv2.imread(image_path)
                    image_resized = cv2.resize(image, (128, 128))
                    image_array = img_to_array(image_resized)
                    data.append(image_array)
                    labels.append(class_name)
                    
                    save_path = os.path.join(class_dest_dir, file)
                    cv2.imwrite(save_path, image_resized)
                except (cv2.error, IOError) as e:
                    print(f"Error processing file {file}: {e}")
                    continue

        process_and_save_files(files, processed_dir)
        print(f"Classe {class_name}: {len(files)} processed")

    data = np.array(data, dtype="float32")
    labels = np.array(labels)

    return data, labels


def increase_dataset(data, labels, zoom_range=0.2, horizontal_flip=True, augmentation_ratio=1/3):
    """
    Augmente la taille d'un dataset d'images en appliquant des transformations 
    d'augmentation telles que le zoom et le flip horizontal.

    Arguments:
    data -- Numpy array contenant les images d'origine.
    labels -- Numpy array contenant les étiquettes correspondantes des images.
    zoom_range -- Intervalle pour le zoom aléatoire pendant l'augmentation 
    (par défaut 0.2).
    horizontal_flip -- Indicateur de flip horizontal des images 
    (par défaut True).
    augmentation_ratio -- Proportion des données à augmenter par rapport à 
    l'ensemble de données d'origine (par défaut 1/3).

    Retourne:
    final_imgs_data -- Numpy array contenant les images originales combinées 
    avec les images augmentées.
    final_labels_data -- Numpy array contenant les étiquettes originales 
    combinées avec les étiquettes augmentées.
    """
    datagen = ImageDataGenerator(
        zoom_range=zoom_range, 
        horizontal_flip=horizontal_flip
    )

    augmented_images = []
    augmented_labels = []

    augmentation_count = int(len(data) * augmentation_ratio)

    for img, label in zip(data, labels):
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        
        i = 0
        for batch in datagen.flow(img, batch_size=1):
            augmented_images.append(batch[0].astype('uint8'))
            augmented_labels.append(label)
            i += 1
            if i >= (augmentation_count / len(data)):
                break

    augmented_images = np.array(augmented_images)
    augmented_labels = np.array(augmented_labels)

    final_imgs_data = np.concatenate((data, augmented_images), axis=0)
    final_labels_data = np.concatenate((labels, augmented_labels), axis=0)

    return final_imgs_data, final_labels_data


def split_data(data, labels, val_size=0.2, test_size=0.2, random_state=42):
    """
    Divise les données en ensembles d'entraînement, de validation et de test.

    Arguments:
    data -- Numpy array contenant les images.
    labels -- Numpy array contenant les étiquettes correspondantes des images.
    val_size -- Proportion des données à utiliser pour la validation 
    (par défaut 0.2).
    test_size -- Proportion des données à utiliser pour le test
     (par défaut 0.2).
    random_state -- Seed pour le générateur de nombres aléatoires 
    (par défaut 42).

    Retourne:
    x_train -- Données d'entraînement.
    x_val -- Données de validation.
    x_test -- Données de test.
    y_train -- Étiquettes d'entraînement.
    y_val -- Étiquettes de validation.
    y_test -- Étiquettes de test.
    """
    x_train_val, x_test, y_train_val, y_test = train_test_split(
        data, 
        labels, 
        test_size=test_size, 
        random_state=random_state
    )

    val_size_adj = val_size / (1 - test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_val, 
        y_train_val, 
        test_size=val_size_adj, 
        random_state=random_state
    )

    return x_train, x_val, x_test, y_train, y_val, y_test


#******************
def check_class_distribution(y_data, classes):
    """
    Vérifie la présence de chaque classe dans l'ensemble de données.

    Arguments:
    y_data -- Étiquettes de l'ensemble de données.
    classes -- Liste des classes à vérifier.

    Retourne:
    Un dictionnaire avec la présence de chaque classe.
    """
    class_distribution = {cls: np.sum(np.argmax(y_data, axis=1) == cls) for cls in classes}
    return class_distribution

