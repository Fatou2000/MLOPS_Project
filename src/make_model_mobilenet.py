""" Modules for make MobileNetV2 model"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


class SparseCategoricalAccuracy(tf.keras.metrics.Metric):
    """Custom class to compute accuracy for sparse categorical labels."""

    def __init__(self, name='Accuracy', **kwargs):
        super().__init__(name=name, **kwargs)
        self.accuracy = tf.keras.metrics.Accuracy()

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Update the state of the metric."""
        if len(y_true.shape) > 1:
            y_true = tf.squeeze(y_true, axis=-1)
        y_pred_classes = tf.argmax(y_pred, axis=-1)
        self.accuracy.update_state(y_true, y_pred_classes, sample_weight)

    def result(self):
        """Compute and return the metric result."""
        return self.accuracy.result()

    def reset_state(self):
        """Reset the metric state."""
        self.accuracy.reset_state()


class SparseCategoricalPrecision(tf.keras.metrics.Metric):
    """Custom class to compute precision for sparse categorical labels."""

    def __init__(self, name='Precision', **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Update the state of the metric."""
        if len(y_true.shape) > 1:
            y_true = tf.squeeze(y_true, axis=-1)
        y_pred_classes = tf.argmax(y_pred, axis=-1)
        self.precision.update_state(y_true, y_pred_classes, sample_weight)

    def result(self):
        """Compute and return the metric result."""
        return self.precision.result()

    def reset_state(self):
        """Reset the metric state."""
        self.precision.reset_state()


class SparseCategoricalRecall(tf.keras.metrics.Metric):
    """Custom class to compute recall for sparse categorical labels."""

    def __init__(self, name='Recall', **kwargs):
        super().__init__(name=name, **kwargs)
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Update the state of the metric."""
        if len(y_true.shape) > 1:
            y_true = tf.squeeze(y_true, axis=-1)
        y_pred_classes = tf.argmax(y_pred, axis=-1)
        self.recall.update_state(y_true, y_pred_classes, sample_weight)

    def result(self):
        """Compute and return the metric result."""
        return self.recall.result()

    def reset_state(self):
        """Reset the metric state."""
        self.recall.reset_state()


class SparseCategoricalAUC(tf.keras.metrics.Metric):
    """Custom class to compute the AUC for sparse categorical labels."""

    def __init__(self, name='AUC', **kwargs):
        super().__init__(name=name, **kwargs)
        self.auc = tf.keras.metrics.AUC(multi_label=True)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Update the state of the metric."""
        if len(y_true.shape) > 1:
            y_true = tf.squeeze(y_true, axis=-1)
        y_true_one_hot = tf.one_hot(tf.cast(y_true, tf.int32), depth=tf.shape(y_pred)[-1])
        self.auc.update_state(y_true_one_hot, y_pred, sample_weight)

    def result(self):
        """Compute and return the metric result."""
        return self.auc.result()

    def reset_state(self):
        """Reset the metric state."""
        self.auc.reset_state()


def create_mobilenetv2_model(input_shape, num_classes, learning_rate):
    """
    Creates and compiles a classification model based on MobileNetV2 with custom metrics.

    Arguments:
    input_shape -- Shape of the input images (height, width, channels).
    num_classes -- Number of classes for classification.
    learning_rate -- Learning rate for the Adam optimizer.

    Returns:
    model -- The compiled Keras model.
    params -- Dictionary containing the parameters used to configure the model.
    """
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=SparseCategoricalCrossentropy(),
        metrics=[
            SparseCategoricalAccuracy(),
            SparseCategoricalPrecision(),
            SparseCategoricalRecall(),
            SparseCategoricalCrossentropy(name='log_loss'),
            SparseCategoricalAUC()
        ]
    )

    params = {
        'input_shape': input_shape,
        'num_classes': num_classes,
        'learning_rate': learning_rate,
        'loss': 'SparseCategoricalCrossentropy',
        'metrics': ['Accuracy', 'Precision', 'Recall', 'log_loss', 'AUC']
    }

    return model, params


def prepare_labels_for_mobilenetv2(labels):
    """
    Prepares labels for training with the MobileNetV2 model.

    If the labels are in one-hot format, they are converted to integers representing the class.

    Arguments:
    labels -- Labels to prepare, in numpy array format.

    Returns:
    labels -- Labels converted to integers, ready for use in training.
    """
    if len(labels.shape) > 1 and labels.shape[-1] > 1:
        # Convert one-hot labels to integers
        labels = np.argmax(labels, axis=-1)
    return labels.astype(np.int32)


def train_mobilenetv2_model(model, x_train, y_train, x_val, y_val, epochs, patience):
    """
    Entraîne un modèle MobileNetV2 en utilisant les données d'entraînement et de validation.

    Arguments:
    model -- Le modèle Keras à entraîner.
    x_train -- Les données d'entraînement (images).
    y_train -- Les étiquettes d'entraînement.
    x_val -- Les données de validation (images).
    y_val -- Les étiquettes de validation.
    epochs -- Le nombre d'époques pour l'entraînement.
    patience -- Le nombre d'époques sans amélioration avant d'arrêter l'entraînement (pour EarlyStopping).

    Retourne:
    history -- L'historique de l'entraînement, contenant des informations sur la performance du modèle à chaque époque.
    """
    # Préparer les étiquettes
    y_train = prepare_labels_for_mobilenetv2(y_train)
    y_val = prepare_labels_for_mobilenetv2(y_val)

    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

    history = model.fit(x_train, y_train,
                        epochs=epochs,
                        validation_data=(x_val, y_val),
                        callbacks=[early_stopping])

    return history



def test_mobilenetv2_model(model, x_test, y_test, class_names):
    """
    Teste le modèle sur toutes les images de l'ensemble de test.

    Arguments:
    model -- Le modèle entraîné.
    x_test -- Données de test.
    y_test -- Étiquettes de test.
    class_names -- Noms des classes.

    Retourne:
    results -- Une liste de tuples contenant la classe prédite et la classe réelle pour chaque image.
    y_pred_proba -- Probabilités prédites pour chaque classe.
    """
    y_test = prepare_labels_for_mobilenetv2(y_test)
    results = []
    y_pred_proba = []

    for index_test, image in enumerate(x_test):
        image_to_predict = np.expand_dims(image, axis=0)
        predictions = model.predict(image_to_predict)
        predicted_class_index = np.argmax(predictions[0])
        real_class_index = y_test[index_test]

        y_pred_proba.append(predictions[0])

        predicted_class = class_names[predicted_class_index]
        real_class = class_names[real_class_index]

        results.append((predicted_class, real_class))

    return results, np.array(y_pred_proba)