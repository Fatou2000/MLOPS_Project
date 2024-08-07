import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from keras.models import load_model

# Chargez votre modèle sauvegardé une seule fois
@st.cache_resource
def load_trained_model():
    return load_model('models/model_best.keras')

model = load_trained_model()

# Classes de déchets
classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

def predict_image(image):
    """
    Prédire la classe de l'image donnée.
    """
    # Redimensionner l'image à la taille requise par le modèle
    image = image.resize((128, 128))
    # Convertir l'image en tableau numpy et normaliser les valeurs des pixels
    image = np.array(image) / 255.0
    # Ajouter une dimension pour correspondre à la forme d'entrée du modèle
    image = np.expand_dims(image, axis=0)
    # Faire la prédiction
    predictions = model.predict(image)
    # Obtenir la classe prédite
    predicted_class = classes[np.argmax(predictions)]
    return predicted_class

# Utilisation des colonnes pour l'affichage
col1, col2 = st.columns([8, 1])  # La colonne de gauche occupe 8 parts, celle de droite 1 part

with col1:
    # Titre de l'application avec emoji
    st.title("♻️ Classification d'ordures")

with col2:
    # Afficher le logo de l'entreprise
    logo = Image.open('logo.png')
    st.image(logo, width=100)  # Ajustez la largeur selon vos préférences


# Sous-titre avec emoji
st.write("🔍 Chargez une image d'ordure et le modèle prédit sa classe.")

# Téléchargement de l'image par l'utilisateur
uploaded_file = st.file_uploader("📥 Choisissez une image...", type=["jpg", "jpeg", "png", "jfif"])

if uploaded_file is not None:
    # Utilisation des colonnes pour l'affichage
    col1, col2 = st.columns([1, 2])

    with col1:
        # Afficher l'image téléchargée avec un cadre et une légende
        image = Image.open(uploaded_file)
        st.image(image, caption='Image téléchargée', use_column_width=True)

    with col2:
        st.write("## 🏷️ Classification en cours...")
        # Faire la prédiction et afficher le résultat
        predicted_class = predict_image(image)
        st.write(f"### La classe prédite est : **{predicted_class}**")

    # Ajouter un bouton pour effacer l'image et la prédiction
    if st.button('🗑️ Effacer et charger une nouvelle image'):
        st.experimental_rerun()
