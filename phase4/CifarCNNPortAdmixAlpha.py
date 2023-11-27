import torch
from attacks.modifiedAdmix import AdmixAttackModified
from utils.datasets import get_dataset
from utils.attackUtils import filterDataSetForTwoModels
from utils.loadModels import getPreTrainedModel
from utils.others import computeAcc
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import numpy as np
print(device)

cifarDS = get_dataset('cifar10', 'test')
loaderCifar = torch.utils.data.DataLoader(cifarDS, batch_size=128, shuffle=False, num_workers=2)
cifarCNN = getPreTrainedModel('CNN', 'cifar10')
print("starting experiment 3 CNN ->  ResNet18PORT")
cifarPORT = getPreTrainedModel('ResNet18-PORT', 'cifar10').to(device)
exp3filteredDS = filterDataSetForTwoModels(cifarCNN, cifarPORT, loaderCifar, False, False)
floader = torch.utils.data.DataLoader(exp3filteredDS, batch_size=128, shuffle=False, num_workers=2)

epsilons = [8, 12, 16, 24, 32, 48, 80]

admixAttackM = AdmixAttackModified(model=cifarCNN, image_width=32, image_height=32)
alphas = [0, 0.125, 0.25, 1, 2, 25, 50, 100]
print("starting now")
for eps in epsilons:
    print("eps=", eps)
    admixBBoxRobustAccList = []
    admixWBoxRobustAccList = []
    for alpha in alphas:
        admixCorrectWB = 0
        admixCorrectBB = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(floader):
            inputs, targets = inputs.to(device), targets.to(device)
            admixAdv = admixAttackM.runAttack(inputs, targets, eps=eps / 255, alphaL2=alpha)
            admixPest = admixAdv - inputs
            # White Box Scenario
            correctAdmixAdvWB, batch_size = computeAcc(cifarCNN, admixAdv, targets)
            admixCorrectWB += correctAdmixAdvWB
            # Black Box Scenario
            correctAdmixAdvBB, batch_size = computeAcc(cifarPORT, admixAdv, targets, is_Vt=False)
            admixCorrectBB += correctAdmixAdvBB

            total += batch_size
        admixAccWhiteBox = 100. * admixCorrectWB / total
        admixAccBlackBox = 100. * admixCorrectBB / total

        admixWBoxRobustAccList.append(admixAccWhiteBox)
        admixBBoxRobustAccList.append(admixAccBlackBox)
    print('admixBlackBoxRobustAccList=', admixBBoxRobustAccList)
    print('admixWhiteBoxRobustAccList=', admixWBoxRobustAccList)
