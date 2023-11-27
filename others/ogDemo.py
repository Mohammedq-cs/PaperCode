from utils.loadOGModels import getOGPreTrainedModel
from utils.datasets import get_dataset

imagenetteResNet = getOGPreTrainedModel('ResNet18', 'imagenette')
imagenetteVt = getOGPreTrainedModel('Visual Transformer', 'imagenette')

cifarVt = getOGPreTrainedModel('Visual Transformer', 'cifar10')
cifarResNet = getOGPreTrainedModel('ResNet18', 'cifar10')

print("hello")
