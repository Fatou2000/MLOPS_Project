# Configuration for VGG19
VGG16_CONFIG = {
    'input_shape': (128, 128, 3),
    'num_classes': 6,
    'learning_rate': 1e-5,
    'batch_size': 32,
    'epochs': 30,
    'pretrained': True,
    'model_path': 'vgg19_pretrained.pth'
}

# Configuration for MobileNetV2
MOBILENETV2_CONFIG = {
    'input_shape': (128, 128, 3),
    'num_classes': 6,
    'learning_rate': 0.0001,
    'epochs':50,
    'patience':5,
    'pretrained': True,
    'model_path': 'mobilenetv2_pretrained.pth'
}

# Configuration for DenseNet
DENSENET_CONFIG = {
    'num_classes': 6,
    'learning_rate': 1e-5,
    'batch_size': 32,
    'epochs': 20,
    'pretrained': True,
    'model_path': 'densenet_pretrained.pth'
}
