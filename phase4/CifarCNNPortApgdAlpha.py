import torch
from utils.datasets import get_dataset
from attacks.modifiedAutoPGD import APGDAttackModified
from utils.attackUtils import filterDataSetForTwoModels
from utils.loadModels import getPreTrainedModel
from utils.others import computeAcc
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

cifarDS = get_dataset('cifar10', 'test')
loaderCifar = torch.utils.data.DataLoader(cifarDS, batch_size=128, shuffle=False, num_workers=2)
cifarCNN = getPreTrainedModel('CNN', 'cifar10')
print("starting experiment 3 CNN ->  ResNet18PORT")
cifarPORT = getPreTrainedModel('ResNet18-PORT', 'cifar10').to(device)
exp3filteredDS = filterDataSetForTwoModels(cifarCNN, cifarPORT, loaderCifar, False, False)
floader = torch.utils.data.DataLoader(exp3filteredDS, batch_size=128, shuffle=False, num_workers=2)

epsilons = [8, 12, 16, 24, 32, 48, 80]

alphas = [0, 1, 2, 8, 16, 32, 64]
print("starting now")
for eps in epsilons:
    print("eps=", eps)
    apgdBBoxRobustAccList = []
    apgdWBoxRobustAccList = []
    for alpha in alphas:
        apgdM = APGDAttackModified(predict=cifarCNN, eps=eps / 255, device=device, alphaL2=alpha, is_Vt=False)
        apgdCorrectWB = 0
        apgdCorrectBB = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(floader):
            inputs, targets = inputs.to(device), targets.to(device)
            apgdAdv = apgdM.perturb(inputs, targets)
            apgdPest = apgdAdv - inputs
            # White Box Scenario
            correctApgdAdvWB, batch_size = computeAcc(cifarCNN, apgdAdv, targets)
            apgdCorrectWB += correctApgdAdvWB
            # Black Box Scenario
            correctApgdAdvBB, batch_size = computeAcc(cifarPORT, apgdAdv, targets, is_Vt=False)
            apgdCorrectBB += correctApgdAdvBB

            total += batch_size
        ApgdAccWhiteBox = 100. * apgdCorrectWB / total
        ApgdAccBlackBox = 100. * apgdCorrectBB / total

        apgdWBoxRobustAccList.append(ApgdAccWhiteBox)
        apgdBBoxRobustAccList.append(ApgdAccBlackBox)
    print('ApgdBlackBoxRobustAccList=', apgdBBoxRobustAccList)
    print('ApgdWhiteBoxRobustAccList=', apgdWBoxRobustAccList)