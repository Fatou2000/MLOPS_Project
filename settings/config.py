# Configuration for VGG16
VGG16_CONFIG = {
    'input_shape': (128, 128, 3),
    'num_classes': 6,
    'learning_rate': 1e-5,
    'batch_size': 32,
    'pretrained' : True,
    'epochs': 50
}

# Configuration for MobileNetV2
MOBILENETV2_CONFIG = {
    'input_shape': (128, 128, 3),
    'num_classes': 6,
    'learning_rate': 0.0001,
    'patience':5,
    'epochs':50
}

# Configuration for DenseNet
DENSENET_CONFIG = {
    'num_classes': 6,
    'dropout_rate': 0.5,
    'min_lr': 1e-8,
    'monitor':'val_accuracy',
    'factor' : 0.15,
    'patience' : 6,
    'learning_rate': 1e-3,
    'batch_size': 64,
    'epochs': 50
}