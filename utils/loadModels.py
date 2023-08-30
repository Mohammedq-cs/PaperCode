import torch
import torchvision.transforms as transforms
from os import path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from models.resnet18Cifar10 import ResNetCifar, BasicBlock
from models.resnet18MNIST import ResNetMNIST, BasicBlock
from models.resnet18imagenette import ResNetImageNette, BasicBlock
from models.LC import LinearClassifier
from models.sCNN import SimpleCNN
from models.swin import SwinTransformer
from models.resnet20Cifar10 import ResNet20Cifar
from models.resnet20Imagenette import ResNet20Imagenette
from models.resnet20MNIST import ResNet20MNIST
from models.ensemble import Ensemble

_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STDDEV = [0.2023, 0.1994, 0.2010]
cifarNormalization = transforms.Normalize(_CIFAR10_MEAN, _CIFAR10_STDDEV)

_imagenette_MEAN = [0.4636, 0.4525, 0.4231]
_imagenette_STDDEV = [0.2216, 0.2156, 0.2204]
imagenette_normalize_layer = transforms.Normalize(_imagenette_MEAN, _imagenette_STDDEV)

_MNIST_MEAN = [0.5, ]
_MNIST_STDDEV = [0.5, ]
mnist_normalize_layer = transforms.Normalize(_MNIST_MEAN, _MNIST_STDDEV)


def getPreTrainedModel(architecture, dataset, models_dir='trained-models'):
    if architecture in model_mapping:
        return model_mapping[architecture](dataset, models_dir)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")


def vtModels(dataset, models_dir):
    DROP_RATE = 0.0
    # Drop path rate
    DROP_PATH_RATE = 0.1
    # Label Smoothing
    LABEL_SMOOTHING = 0.1

    # Swin Transformer parameters
    NUM_CLAAEA = 10
    PATCH_SIZE = 4
    IN_CHANS = 3
    EMBED_DIM = 96
    DEPTHS = [2, 2, 6, 2]
    NUM_HEADS = [3, 6, 12, 24]
    WINDOW_SIZE = 7
    MLP_RATIO = 4.
    QKV_BIAS = True
    QK_SCALE = None
    APE = False
    RPE = True
    PATCH_NORM = True
    IMG_SIZE = 224
    transform_test = torch.nn.Sequential(
        transforms.Resize(size=(256, 256), interpolation=2, max_size=None, antialias='warn'),
        transforms.CenterCrop(size=(224, 224)),
        transforms.Normalize(mean=torch.tensor([0.4850, 0.4560, 0.4060]), std=torch.tensor([0.2290, 0.2240, 0.2250]))
    )
    transform_test_mnist = torch.nn.Sequential(
        transforms.Resize(IMG_SIZE, interpolation=transforms.InterpolationMode.BICUBIC)
    )
    if dataset == 'cifar10':
        fpath = path.join(models_dir, f'cifar10-vt.pth')
        cifar10Net = eval(
            'SwinTransformer(img_size = 224, patch_size=PATCH_SIZE, num_classes= 10, embed_dim= EMBED_DIM, depths= DEPTHS, num_heads= NUM_HEADS, '
            'window_size=WINDOW_SIZE, mlp_ratio= MLP_RATIO, qkv_bias= QKV_BIAS, qk_scale= QK_SCALE, drop_rate= DROP_RATE, drop_path_rate= '
            'DROP_PATH_RATE)').to(
            device)
        cifar10Net = torch.nn.DataParallel(cifar10Net)
        cifar10Net.load_state_dict((torch.load(fpath, map_location=device)))
        cifar10Net = torch.nn.Sequential(transform_test, cifar10Net)
        return cifar10Net
    elif dataset == 'mnist':
        fpath = path.join(models_dir, f'mnist-vt.pth')
        MNISTnet = eval(
            'SwinTransformer(img_size= 224, in_chans = 1, patch_size=PATCH_SIZE, num_classes= 10, embed_dim= EMBED_DIM, depths= DEPTHS, '
            'num_heads= NUM_HEADS, window_size=WINDOW_SIZE, mlp_ratio= MLP_RATIO, qkv_bias= QKV_BIAS, qk_scale= QK_SCALE, drop_rate= DROP_RATE, '
            'drop_path_rate= DROP_PATH_RATE)').to(
            device)
        MNISTnet = torch.nn.DataParallel(MNISTnet)
        MNISTnet.load_state_dict((torch.load(fpath, map_location=device)))
        MNISTnet = torch.nn.Sequential(transform_test_mnist, MNISTnet)
        return MNISTnet
    elif dataset == 'imagenette':
        fpath = path.join(models_dir, f'imagenette-vt.pth')
        imagenetteNet = eval(
            'SwinTransformer(img_size= IMG_SIZE, patch_size=PATCH_SIZE, num_classes= 10, embed_dim= EMBED_DIM, depths= DEPTHS, '
            'num_heads= NUM_HEADS, window_size=WINDOW_SIZE, mlp_ratio= MLP_RATIO, qkv_bias= QKV_BIAS, qk_scale= QK_SCALE, drop_rate= DROP_RATE, '
            'drop_path_rate= DROP_PATH_RATE)').to(
            device)
        imagenetteNet = torch.nn.DataParallel(imagenetteNet)
        imagenetteNet.load_state_dict(torch.load(fpath, map_location=device))
        imagenetteNet = torch.nn.Sequential(transform_test, imagenetteNet)
        return imagenetteNet


def lcModels(dataset, models_dir):
    if dataset == 'cifar10':
        fpath = path.join(models_dir, f'LinearClassifierCifar10.pth')
        cifarLC = eval(f'LinearClassifier(3 * 32 * 32, 10)').to(device)
        cifarLC.load_state_dict(torch.load(fpath, map_location=device))
        cifarLC = torch.nn.Sequential(cifarNormalization, cifarLC).to(device)
        return cifarLC
    elif dataset == 'mnist':
        fpath = path.join(models_dir, f'LinearClassifierMNIST.pth')
        mnistLC = eval(f'LinearClassifier(1 * 28 * 28, 10)').to(device)
        mnistLC.load_state_dict(torch.load(fpath, map_location=device))
        return mnistLC
    elif dataset == 'imagenette':
        fpath = path.join(models_dir, f'LinearClassifierImagenette.pth')
        imageNetteLC = eval(f'LinearClassifier(3 * 160 * 160, 10)').to(device)
        imageNetteLC.load_state_dict(torch.load(fpath, map_location=device))
        return imageNetteLC


def resnet18Models(dataset, models_dir):
    if dataset == 'cifar10':
        fpath = path.join(models_dir, f'resnet18Cifar10.pth')
        cifarResnet = eval(f'ResNetCifar(BasicBlock, [2,2,2,2])').to(device)
        cifarResnet = torch.nn.DataParallel(cifarResnet)
        cifarResnet.load_state_dict(torch.load(fpath, map_location=device))
        cifarResnet = torch.nn.Sequential(cifarNormalization, cifarResnet).to(device)
        return cifarResnet
    elif dataset == 'mnist':
        fpath = path.join(models_dir, f'resnet18Minist.pth')
        mnistResnet = eval(f'ResNetMNIST(BasicBlock, [2,2,2,2])').to(device)
        mnistResnet = torch.nn.DataParallel(mnistResnet)
        mnistResnet.load_state_dict(torch.load(fpath, map_location=device))
        return mnistResnet
    elif dataset == 'imagenette':
        fpath = path.join(models_dir, f'Resnetl8Imagenette.pth')
        imageNetteResnet = eval(f'ResNetImageNette(BasicBlock, [2,2,2,2])').to(device)
        imageNetteResnet = torch.nn.DataParallel(imageNetteResnet)
        imageNetteResnet.load_state_dict(torch.load(fpath, map_location=device))
        return imageNetteResnet


def cnnModels(dataset, models_dir):
    if dataset == 'cifar10':
        fpath = path.join(models_dir, f'sCNN-CIFAR10.pth')
        cifarSCNN = eval(f'SimpleCNN(inDim = 3, flattenLayerValue =  8 * 8)').to(device)
        cifarSCNN.load_state_dict(torch.load(fpath, map_location=device))
        cifarSCNN = torch.nn.Sequential(cifarNormalization, cifarSCNN).to(device)
        return cifarSCNN
    elif dataset == 'mnist':
        fpath = path.join(models_dir, f'sCNN-MNIST.pth')
        mnistSCNN = eval(f'SimpleCNN(inDim = 1, flattenLayerValue =  7 * 7)').to(device)
        mnistSCNN.load_state_dict(torch.load(fpath, map_location=device))
        return mnistSCNN
    elif dataset == 'imagenette':
        fpath = path.join(models_dir, f'sCNN-IMAGENETTE.pth')
        imageNetteSCNN = eval(f'SimpleCNN(inDim = 3, flattenLayerValue = 40*40)').to(device)
        imageNetteSCNN.load_state_dict(torch.load(fpath, map_location=device))
        return imageNetteSCNN


def resnet20TRSModels(dataset, models_dir):
    if dataset == 'cifar10':
        fpath1 = path.join(models_dir, f'CIFAR10-TRS-0.pth')
        fpath2 = path.join(models_dir, f'CIFAR10-TRS-1.pth')
        fpath3 = path.join(models_dir, f'CIFAR10-TRS-2.pth')

        # load ensemble models
        cifarModel1 = eval(f'ResNet20Cifar(depth=20, num_classes=10)').to(device)
        cifarModel1 = torch.nn.Sequential(cifarNormalization, cifarModel1).to(device)
        cifarModel1 = torch.nn.DataParallel(cifarModel1)

        # load ensemble models
        cifarModel2 = eval(f'ResNet20Cifar(depth=20, num_classes=10)').to(device)
        cifarModel2 = torch.nn.Sequential(cifarNormalization, cifarModel2).to(device)
        cifarModel2 = torch.nn.DataParallel(cifarModel2)

        # load ensemble models
        cifarModel3 = eval(f'ResNet20Cifar(depth=20, num_classes=10)').to(device)
        cifarModel3 = torch.nn.Sequential(cifarNormalization, cifarModel3).to(device)
        cifarModel3 = torch.nn.DataParallel(cifarModel3)

        cifarModel1.load_state_dict(torch.load(fpath1, map_location=device))
        cifarModel2.load_state_dict(torch.load(fpath2, map_location=device))
        cifarModel3.load_state_dict(torch.load(fpath3, map_location=device))

        cifarEnsemble = Ensemble([cifarModel1, cifarModel2, cifarModel3])
        return cifarEnsemble
    elif dataset == 'mnist':
        fpath1 = path.join(models_dir, f'MNIST-TRS-0.pth')
        fpath2 = path.join(models_dir, f'MNIST-TRS-1.pth')
        fpath3 = path.join(models_dir, f'MNIST-TRS-2.pth')

        MNISTModel1 = eval(f'ResNet20MNIST(depth = 20)').to(device)
        MNISTModel1 = torch.nn.Sequential(mnist_normalize_layer, MNISTModel1).to(device)
        MNISTModel1 = torch.nn.DataParallel(MNISTModel1)

        MNISTModel2 = eval(f'ResNet20MNIST(depth = 20)').to(device)
        MNISTModel2 = torch.nn.Sequential(mnist_normalize_layer, MNISTModel2).to(device)
        MNISTModel2 = torch.nn.DataParallel(MNISTModel2)

        MNISTModel3 = eval(f'ResNet20MNIST(depth = 20)').to(device)
        MNISTModel3 = torch.nn.Sequential(mnist_normalize_layer, MNISTModel3).to(device)
        MNISTModel3 = torch.nn.DataParallel(MNISTModel3)

        MNISTModel1.load_state_dict(torch.load(fpath1, map_location=device))
        MNISTModel2.load_state_dict(torch.load(fpath2, map_location=device))
        MNISTModel3.load_state_dict(torch.load(fpath3, map_location=device))

        mnistEnsemble = Ensemble([MNISTModel1, MNISTModel2, MNISTModel3])
        return mnistEnsemble

    elif dataset == 'imagenette':
        fpath1 = path.join(models_dir, f'IMAGENETTE-TRS-0.pth')
        fpath2 = path.join(models_dir, f'IMAGENETTE-TRS-1.pth')
        fpath3 = path.join(models_dir, f'IMAGENETTE-TRS-2.pth')

        ImagenetteModel1 = eval(f'ResNet20Imagenette(depth = 20)').to(device)
        ImagenetteModel1 = torch.nn.Sequential(imagenette_normalize_layer, ImagenetteModel1).to(device)
        ImagenetteModel1 = torch.nn.DataParallel(ImagenetteModel1)

        ImagenetteModel2 = eval(f'ResNet20Imagenette(depth = 20)').to(device)
        ImagenetteModel2 = torch.nn.Sequential(imagenette_normalize_layer, ImagenetteModel2).to(device)
        ImagenetteModel2 = torch.nn.DataParallel(ImagenetteModel2)

        ImagenetteModel3 = eval(f'ResNet20Imagenette(depth = 20)').to(device)
        ImagenetteModel3 = torch.nn.Sequential(imagenette_normalize_layer, ImagenetteModel3).to(device)
        ImagenetteModel3 = torch.nn.DataParallel(ImagenetteModel3)

        ImagenetteModel1.load_state_dict(torch.load(fpath1, map_location=device))
        ImagenetteModel2.load_state_dict(torch.load(fpath2, map_location=device))
        ImagenetteModel3.load_state_dict(torch.load(fpath3, map_location=device))

        imagenetteEnsemble = Ensemble([ImagenetteModel1, ImagenetteModel2, ImagenetteModel3])
        return imagenetteEnsemble


def resnet18PORTModels(dataset, models_dir):
    if dataset == 'cifar10':
        fpath = path.join(models_dir, f'Cifar10-ResNet18-Port.pth')
        cifarResnetTRS = eval(f'ResNetCifar(BasicBlock, [2,2,2,2])')
        cifarResnetTRS.load_state_dict(torch.load(fpath, map_location=device))
        return cifarResnetTRS
    else:
        raise ValueError(f"yet to be implemented")


model_mapping = {
    'Visual Transformer': vtModels,
    'ResNet18': resnet18Models,
    'Linear Classifier': lcModels,
    'CNN': cnnModels,
    'ResNet20-TRSEnsemble': resnet20TRSModels,
    'ResNet18-PORT': resnet18PORTModels
}
