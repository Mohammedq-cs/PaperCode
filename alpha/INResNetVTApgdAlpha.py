import torch
from utils.datasets import get_dataset
from attacks.modifiedAutoPGD import APGDAttackModified
from utils.attackUtils import filterDataSetForTwoModels
from utils.loadModels import getPreTrainedModel
from utils.others import computeAcc
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

imagenetteDS = get_dataset('imagenette', 'test')
loaderIN = torch.utils.data.DataLoader(imagenetteDS, batch_size=60, shuffle=False, num_workers=2)

imagenetteResNet = getPreTrainedModel('ResNet18', 'imagenette')
imagenetteVt = getPreTrainedModel('Visual Transformer', 'imagenette')

filteredDS = filterDataSetForTwoModels(imagenetteResNet, imagenetteVt, loaderIN, False, True)
floader = torch.utils.data.DataLoader(filteredDS, batch_size=60, shuffle=False, num_workers=2)

epsilons = [8, 12, 16, 24, 32, 48, 80]

alphas = [0, 1, 2, 8, 16, 32, 64]
print("starting now")
for eps in epsilons:
    print("eps=", eps)
    apgdBBoxRobustAccList = []
    apgdWBoxRobustAccList = []
    for alpha in alphas:
        apgdM = APGDAttackModified(predict=imagenetteResNet, eps=eps / 255, device=device, alphaL2=alpha, is_Vt=False)
        apgdCorrectWB = 0
        apgdCorrectBB = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(floader):
            inputs, targets = inputs.to(device), targets.to(device)
            apgdAdv = apgdM.perturb(inputs, targets)
            apgdPest = apgdAdv - inputs
            # White Box Scenario
            correctApgdAdvWB, batch_size = computeAcc(imagenetteResNet, apgdAdv, targets)
            apgdCorrectWB += correctApgdAdvWB
            # Black Box Scenario
            correctApgdAdvBB, batch_size = computeAcc(imagenetteVt, apgdAdv, targets, is_Vt=True)
            apgdCorrectBB += correctApgdAdvBB

            total += batch_size
        ApgdAccWhiteBox = 100. * apgdCorrectWB / total
        ApgdAccBlackBox = 100. * apgdCorrectBB / total

        apgdWBoxRobustAccList.append(ApgdAccWhiteBox)
        apgdBBoxRobustAccList.append(ApgdAccBlackBox)
    print('ApgdBlackBoxRobustAccList=', apgdBBoxRobustAccList)
    print('ApgdWhiteBoxRobustAccList=', apgdWBoxRobustAccList)