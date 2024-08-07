# Projet de Classification des Ordures

Ce repository contient un projet de classification d'ordures, structuré comme suit :

## Structure du Repository

- **notebooks/** : Ce répertoire contient les différents notebooks Jupyter utilisés pour le projet.
  - `garbage_classification_final.ipynb` : Notebook final pour la classification des ordures. Contient les étapes de prétraitement des données, la construction et l'évaluation des modèles de classification, ainsi que le monitoring des modèles avec MLFlow.

  - `pylint.ipynb` : Notebook pour l'analyse de la qualité du code avec Pylint.

- **src/** : Ce répertoire contient les différentes fonctions et modules utilisés pour le projet.
  - `make_model_densenet.py` : Module pour la création et l'entraînement du modèle DenseNet.

  - `make_model_mobilenet.py` : Module pour la création et l'entraînement du modèle MobileNetV2.

  - `make_model_vgg16.py` : Module pour la création et l'entraînement du modèle VGG16.

  - `make_dataset.py` : Module pour la gestion des données, y compris la séparation en ensembles d'entraînement, de validation et de test.

  - `eval_metrics.py` : Définit une fonction pour évaluer les performances les modèles.

- **settings/** : Ce répertoire contient les configurations des modèles(paramètres pour l'entrainement).
    - `config.py` : Module pour les paramètres des modèles.

- **tests/** : Ce répertoire contient les tests associés aux différentes parties du projet pour assurer la qualité et le bon fonctionnement du code.

- **requirements.txt** : Fichier listant les dépendances nécessaires pour exécuter le projet.

- **run_garbage_classification.sh** : Script shell pour exécuter le projet de classification des ordures.

- **.pylintrc** : Fichier de configuration pour Pylint, utilisé pour l'analyse statique du code.

- **`streamlit_app/`**  

  Contient l'application Streamlit pour tester le modèle de classification.  
  - `app.py` : Script principal de l'application Streamlit. Pour exécuter l'application, utilisez la commande :

    ```bash
    streamlit run app.py

  - Accéder au Modèle Entraîné

Le modèle entraîné est disponible sur Google Drive. Vous pouvez le télécharger en utilisant le lien ci-dessous :

[Télécharger le modèle entraîné (model_best.keras)](https://drive.google.com/file/d/1IkLReT9dNKQiimy8Vl5Oi9MAz0BNAr2v/view?usp=sharing)

### Instructions pour Utiliser le Modèle

1. Téléchargez le modèle à partir du lien ci-dessus.
2. Placez le fichier `model_best.keras` dans le répertoire `models` situé sous `streamlit_app.

## Obtenir le Dataset

Le projet utilise le dataset Garbage Classification de Kaggle pour l'entraînement et la validation du modèle. Vous pouvez le télécharger à partir du lien suivant :

[**Télécharger le Dataset**](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification?datasetId=81794&sortBy=voteCount)

### Structure du Dataset

Le dataset contient des images classifiées en 6 catégories. Voici la répartition des classes :

- **Cardboard**: 403 images
- **Glass**: 501 images
- **Metal**: 410 images
- **Paper**: 594 images
- **Plastic**: 482 images
- **Trash**: 137 images

Chaque image est placée dans un répertoire correspondant à sa classe. Assurez-vous que la structure des dossiers reflète cette distribution pour un traitement correct des données.


## Instructions d'Utilisation

1. **Installation des Dépendances** : 
   Assurez-vous que vous avez toutes les dépendances nécessaires en installant les packages listés dans `requirements.txt` :
   ```bash
   pip install -r requirements.txt

2. **Execution des notebooks** : 

   - **Jupyter** :  
   Pour exécuter les notebooks, vous pouvez utiliser Jupyter Notebook

   - **CLI**:  
  Pour obtenir une version des résultats de l'exécution du notebook, vous pouvez utiliser le fichier `run_garbage_classification.sh`. Ce script utilise la bibliothèque `papermill` pour exécuter le notebook et sauvegarder les résultats dans le dossier `logs` (qui sera automatiquement créé si nécessaire).  

  a. Placez-vous dans le dossier `MLOPS_PROJECT`.  
  b. Exécutez les commandes suivantes :


        sh run_garbage_classification.sh ou ./run_garbage_classification.sh

