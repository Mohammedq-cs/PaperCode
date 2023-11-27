import torch
from utils.datasets import get_dataset
from attacks.modifiedVmi import VMIAttackModified
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

vmiAttackM = VMIAttackModified(model=cifarCNN)
alphas = [0, 10, 100, 1000, 10000, 100000, 500000, 1000000]
print("starting now")
for eps in epsilons:
    print("eps=", eps)
    vmiBBoxRobustAccList = []
    vmiWBoxRobustAccList = []
    for alpha in alphas:
        vmiCorrectWB = 0
        vmiCorrectBB = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(floader):
            inputs, targets = inputs.to(device), targets.to(device)
            vmiAdv = vmiAttackM.runAttack(inputs, targets, eps=eps / 255, alphaL2=alpha)
            vmiPest = vmiAdv - inputs
            # White Box Scenario
            correctVmiAdvWB, batch_size = computeAcc(cifarCNN, vmiAdv, targets)
            vmiCorrectWB += correctVmiAdvWB
            # Black Box Scenario
            correctVmiAdvBB, batch_size = computeAcc(cifarPORT, vmiAdv, targets, is_Vt=False)
            vmiCorrectBB += correctVmiAdvBB

            total += batch_size
        vmiAccWhiteBox = 100. * vmiCorrectWB / total
        vmiAccBlackBox = 100. * vmiCorrectBB / total

        vmiWBoxRobustAccList.append(vmiAccWhiteBox)
        vmiBBoxRobustAccList.append(vmiAccBlackBox)
    print('vmiBlackBoxRobustAccList=', vmiBBoxRobustAccList)
    print('vmiWhiteBoxRobustAccList=', vmiWBoxRobustAccList)