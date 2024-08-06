import unittest
import numpy as np
import sys
import os
from unittest.mock import patch
from sklearn.preprocessing import LabelBinarizer
from sklearn.datasets import load_sample_image
from sklearn.model_selection import train_test_split
# Ajoutez le répertoire src au chemin d'importation
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from make_dataset import load_data, visualize_class_distribution, show_random_images, plot_images_from_subfolders, img_dimensions, process_dataset, increase_dataset, split_data, check_class_distribution

class TestImageProcessing(unittest.TestCase):
    def setUp(self):
        # Configurez un répertoire de test ou utilisez des données fictives
        self.test_dir = 'test_data'
        self.processed_dir = 'processed_data'
        self.classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

        # Generate some sample data for testing split_data and check_class_distribution
        self.sample_data = np.array([load_sample_image('china.jpg') for _ in range(10)])
        self.sample_labels = np.array([0, 1] * 5)  # Simple labels for testing
        self.class_names = ['class1', 'class2']

    def test_load_data(self):
        images, labels, class_names = load_data(self.test_dir)
        self.assertIsInstance(images, np.ndarray)
        self.assertIsInstance(labels, np.ndarray)
        self.assertTrue(class_names)
        self.assertEqual(len(images), len(labels))
        self.assertTrue(set(class_names).issubset(set(self.classes)))

    def test_visualize_class_distribution(self):
        _, labels, class_names = load_data(self.test_dir)
        with patch('matplotlib.pyplot.show'):
            visualize_class_distribution(labels, class_names)

    def test_plot_images_from_subfolders(self):
        with patch('matplotlib.pyplot.show'):
            plot_images_from_subfolders(self.test_dir, num_images=3)

    def test_img_dimensions(self):
        with patch('builtins.print') as mock_print:
            img_dimensions(self.test_dir)
            mock_print.assert_called()

    def test_process_dataset(self):
        data, labels = process_dataset(self.test_dir, self.processed_dir)
        self.assertIsInstance(data, np.ndarray)
        self.assertIsInstance(labels, np.ndarray)
        self.assertEqual(len(data), len(labels))

    def test_increase_dataset(self):
        images, labels, _ = load_data(self.test_dir)
        augmented_data, augmented_labels = increase_dataset(images, labels, augmentation_ratio=0.1)
        self.assertGreater(len(augmented_data), len(images))
        self.assertGreater(len(augmented_labels), len(labels))

    def test_split_data(self):
        x_train, x_val, x_test, y_train, y_val, y_test = split_data(self.sample_data, self.sample_labels)
        self.assertEqual(x_train.shape[0] + x_val.shape[0] + x_test.shape[0], self.sample_data.shape[0])
        self.assertEqual(y_train.shape[0] + y_val.shape[0] + y_test.shape[0], self.sample_labels.shape[0])

    def test_check_class_distribution(self):
        # Classes pour le test, les indices correspondent à la position des classes
        classes = {
            'cardboard': 0,
            'glass': 1,
            'metal': 2,
            'paper': 3,
            'plastic': 4,
            'trash': 5
        }

        # Préparez des données de test one-hot encodées
        # Exemple de y_data pour 4 éléments avec les classes 'cardboard' et 'glass'
        y_data = np.array([
            [1, 0, 0, 0, 0, 0],  # cardboard
            [0, 1, 0, 0, 0, 0],  # glass
            [0, 1, 0, 0, 0, 0],  # glass
            [1, 0, 0, 0, 0, 0],  # cardboard
            [0, 0, 0, 0, 1, 0],  # plastic
            [1, 0, 0, 0, 0, 0],  # cardboard
            [0, 0, 0, 1, 0, 0],  # paper
            [0, 1, 0, 0, 0, 0],  # glass
            [1, 0, 0, 0, 0, 0],  # cardboard  <-- Ajouter une occurrence supplémentaire de cardboard
            [0, 1, 0, 0, 0, 0],  # glass  <-- Ajouter une occurrence supplémentaire de glass
        ])

        # Appeler la fonction de distribution des classes
        distribution = check_class_distribution(y_data, list(classes.values()))

        # Définir les attentes (en fonction de y_data)
        expected_distribution = {
            0: 4,  # cardboard
            1: 4,  # glass
            2: 0,  # metal
            3: 1,  # paper
            4: 1,  # plastic
            5: 0   # trash
        }
        # Vérifiez que la distribution est correcte
        self.assertEqual(distribution, expected_distribution)

if __name__ == '__main__':
    unittest.main()
