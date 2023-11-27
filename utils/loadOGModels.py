import torch
import torchvision.transforms as transforms
from os import path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from ogModels.resnet18Cifar10 import ResNetCifar, BasicBlock
from ogModels.resnet18imagenette import ResNetImageNette, BasicBlock
from ogModels.vitT2T import T2T_ViT
from ogModels.basicCNN import BasicCNN


def getOGPreTrainedModel(architecture, dataset, models_dir='ogPretrained'):
    if architecture in model_mapping:
        return model_mapping[architecture](dataset, models_dir)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")


def vitModels(dataset, models_dir):
    transform_test = torch.nn.Sequential(
        transforms.Resize(size=(224, 224))
    )
    if dataset == 'cifar10':
        fpath = path.join(models_dir, f'og-Cifar-vitT2T.pth')
        cifarViT = eval(
            f'T2T_ViT(img_size=32, num_classes=10, in_chans=3, embed_dim=384, depth=14, num_heads=6, mlp_ratio=3.0, token_dim=64, use_drloc=False, '
            f'sample_size=32, use_abs=False)').to(
            device)
        cifarViT = torch.nn.DataParallel(cifarViT)
        cifarViT.load_state_dict(torch.load(fpath, map_location=device))
        return cifarViT
    elif dataset == 'imagenette':
        fpath = path.join(models_dir, f'og-INette-vitT2T.pth')
        inetteViT = eval(
            f'T2T_ViT(img_size=224, num_classes=10, in_chans=3, embed_dim=384, depth=14, num_heads=6, mlp_ratio=3.0, token_dim=64, use_drloc=False, '
            f'sample_size=32, use_abs=False)').to(device)
        inetteViT = torch.nn.DataParallel(inetteViT)
        inetteViT.load_state_dict(torch.load(fpath, map_location=device))
        inetteViT = torch.nn.Sequential(transform_test, inetteViT).to(device)
        return inetteViT
    else:
        raise ValueError(f"not implemented yet")


def resnet18Models(dataset, models_dir):
    if dataset == 'cifar10':
        fpath = path.join(models_dir, f'og-Cifar-resnet18.pth')
        cifarResnet = eval(f'ResNetCifar(BasicBlock, [2,2,2,2])').to(device)
        cifarResnet = torch.nn.DataParallel(cifarResnet)
        cifarResnet.load_state_dict(torch.load(fpath, map_location=device))
        return cifarResnet
    elif dataset == 'imagenette':
        transform_test = torch.nn.Sequential(
            transforms.Resize(size=(224, 224))
        )
        fpath = path.join(models_dir, f'og-INette-resnet18.pth')
        imageNetteResnet = eval(f'ResNetImageNette(BasicBlock, [2,2,2,2])').to(device)
        imageNetteResnet = torch.nn.DataParallel(imageNetteResnet)
        imageNetteResnet.load_state_dict(torch.load(fpath, map_location=device))
        imageNetteResnet = torch.nn.Sequential(transform_test, imageNetteResnet).to(device)
        return imageNetteResnet
    else:
        raise ValueError(f"not implemented yet")

def resnet18PORTModels(dataset, models_dir):
    if dataset == 'cifar10':
        fpath = path.join(models_dir, f'Cifar10-ResNet18-Port.pth')
        cifarResnetTRS = eval(f'ResNetCifar(BasicBlock, [2,2,2,2])')
        cifarResnetTRS.load_state_dict(torch.load(fpath, map_location=device))
        return cifarResnetTRS.to(device)
    else:
        raise ValueError(f"yet to be implemented")


def cnnModels(dataset, models_dir):
    if dataset == 'cifar10':
        fpath = path.join(models_dir, f'og-BasicCNN-Cifar10.pth')
        basicCNNmodel = eval(f'BasicCNN(in_channels=3, conv_channels=64, fc_nodes=256, fc_nodes_in = 2 * 64 * 5 *5, maxpool_param=2, num_classes=10)')
        basicCNNmodel = torch.nn.DataParallel(basicCNNmodel)
        basicCNNmodel.load_state_dict(torch.load(fpath, map_location=device))
        return basicCNNmodel.to(device)
    else:
        raise ValueError(f"yet to be implemented")


model_mapping = {
        'Visual Transformer': vitModels,
        'ResNet18': resnet18Models,
        'CNN': cnnModels,
        'ResNet18-PORT': resnet18PORTModels
    }

