"""Module pour créer un modèle DenseNet"""

# Importations regroupées en haut du fichier
import os
import shutil
import sys
from keras.applications import DenseNet121
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.layers import BatchNormalization, Dense, Dropout, GlobalAveragePooling2D
from keras.models import Model
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from make_dataset import split_data


def build_densenet_model(config, data, labels, input_shape=(128, 128, 3)):
    """
    Construit un modèle DenseNet121 avec des couches personnalisées pour la classification et
    sépare les données en ensembles d'entraînement, de validation et de test.

    Arguments:
    config -- Dictionnaire contenant les configurations du modèle, incluant le nombre de classes et le taux de dropout.
    data -- Les données d'entrée pour l'entraînement et la validation.
    labels -- Les étiquettes des données correspondantes.
    input_shape -- La forme des images d'entrée (hauteur, largeur, canaux).

    Retourne:
    model -- Le modèle Keras compilé prêt pour l'entraînement.
    x_train -- Les données d'entraînement.
    x_val -- Les données de validation.
    x_test -- Les données de test.
    y_train -- Les étiquettes d'entraînement.
    y_val -- Les étiquettes de validation.
    y_test -- Les étiquettes de test.
    """
    # Chargement du modèle de base DenseNet121
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)

    # Ajout de nouvelles couches pour la classification
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(config['dropout_rate'])(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(config['dropout_rate'])(x)

    # Couche de prédiction finale
    preds = Dense(config['num_classes'], activation='softmax')(x)

    # Création du modèle
    model = Model(inputs=base_model.input, outputs=preds)

    # Compilation du modèle
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=[
        'accuracy', 
        'Precision', 
        'Recall', 
        'AUC'])
    
    # Séparation des données en ensembles d'entraînement, de validation et de test
    x_train, x_val, x_test, y_train, y_val, y_test = split_data(data, labels)

    return model, x_train, x_val, x_test, y_train, y_val, y_test


def train_densenet_model(model, config, x_train, y_train, x_val, y_val, 
                         drive_path='/content/drive/MyDrive/MLOPS_Project'):
    """
    Entraîne le modèle avec les images augmentées et sauvegarde le meilleur modèle dans Google Drive.

    Arguments:
    model -- Le modèle Keras à entraîner.
    x_train -- Les données d'entraînement.
    y_train -- Les étiquettes d'entraînement.
    x_val -- Les données de validation.
    y_val -- Les étiquettes de validation.
    drive_path -- Le chemin vers Google Drive où le modèle sera sauvegardé.

    Retourne:
    history -- L'historique de l'entraînement.
    model_params -- Les paramètres du modèle après l'entraînement.
    """
    # Réduction du taux d'apprentissage
    anne = ReduceLROnPlateau(monitor=config['monitor'], 
                             factor=config['factor'], 
                             patience=config['patience'], 
                             verbose=True, 
                             min_lr=config['min_lr'])
    # Enregistrement des meilleurs modèles
    checkpoint_path = 'model_best.keras'
    checkpoint = ModelCheckpoint(checkpoint_path, verbose=1, save_best_only=True)
    # Génération d'images augmentées
    datagen = ImageDataGenerator(zoom_range=0.2, horizontal_flip=True, shear_range=0.2)
    # Ajustement du générateur d'images aux données d'entraînement
    datagen.fit(x_train)
    # Entrainement du modèle avec les images augmentées
    history = model.fit(datagen.flow(x_train, y_train, batch_size=config['batch_size']),
                        steps_per_epoch=x_train.shape[0] // config['batch_size'],
                        epochs=config['epochs'],
                        verbose=2,
                        callbacks=[anne, checkpoint],
                        validation_data=(x_val, y_val))

    # Copier le meilleur modèle dans Google Drive
    if os.path.exists(checkpoint_path):
        drive_model_path = os.path.join(drive_path, 'model_best.keras')
        shutil.copy(checkpoint_path, drive_model_path)
        print(f"Model saved to Google Drive at {drive_model_path}")
    else:
        print("Best model not found!")

    # Récupérer les paramètres de l'entraînement
    training_params = {
        'batch_size': config['batch_size'],
        'epochs': config['epochs'],
        'steps_per_epoch': x_train.shape[0] // config['batch_size'],
        'learning_rate': model.optimizer.learning_rate.numpy(),
        'monitor': config['monitor'],
        'factor': config['factor'],
        'patience': config['patience'],
        'min_lr': config['min_lr']
    }

    return history, training_params
