# Configuration for VGG19
VGG19_CONFIG = {
    'num_classes': 6,
    'learning_rate': 1e-5,
    'batch_size': 32,
    'epochs': 5,
    'pretrained': True,
    'model_path': 'vgg19_pretrained.pth'
}

# Configuration for MobileNetV2
MOBILENETV2_CONFIG = {
    'num_classes': 6,
    'learning_rate': 1e-5,
    'batch_size': 32,
    'epochs': 20,
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
