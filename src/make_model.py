"""Module for training data"""

from sklearn.model_selection import train_test_split
import numpy as np

def split_data(data, labels, test_size=0.15, val_size=0.15, random_state=42):
    """
    Divise les données en ensembles d'entraînement, de validation et de test.

    Arguments:
    data -- Les données à diviser (numpy array).
    labels -- Les étiquettes correspondantes (numpy array).
    test_size -- Proportion des données à utiliser pour l'ensemble de test.
    val_size -- Proportion des données d'entraînement à utiliser pour l'ensemble de validation.
    random_state -- Graine pour la reproductibilité.

    Retourne:
    x_train, x_val, x_test -- Données divisées en ensembles d'entraînement, de validation et de test.
    y_train, y_val, y_test -- Étiquettes divisées en ensembles d'entraînement, de validation et de test.
    """

    # Diviser les données en ensembles d'entraînement et de test
    x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=test_size, random_state=random_state)

    # Diviser l'ensemble d'entraînement en ensembles d'entraînement et de validation
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_size, random_state=random_state)

    # Affichage des dimensions des ensembles
    print(f"Ensemble d'entraînement : {x_train.shape}, Ensemble de validation : {x_val.shape}, Ensemble de test : {x_test.shape}")

    return x_train, x_val, x_test, y_train, y_val, y_test


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



