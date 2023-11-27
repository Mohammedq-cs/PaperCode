import torch
from attacks.modifiedVmi import VMIAttackModified
from utils.datasets import get_dataset
from utils.attackUtils import filterDataSetForTwoModels
from utils.loadModels import getPreTrainedModel
from utils.others import computeAcc
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

cifarDS = get_dataset('cifar10', 'test')
loaderCifar = torch.utils.data.DataLoader(cifarDS, batch_size=128, shuffle=False, num_workers=2)

cifarVt = getPreTrainedModel('Visual Transformer', 'cifar10')
cifarResNet = getPreTrainedModel('ResNet18', 'cifar10')

filteredDS = filterDataSetForTwoModels(cifarResNet, cifarVt, loaderCifar, False, True)
floader = torch.utils.data.DataLoader(filteredDS, batch_size=128, shuffle=False, num_workers=2)

epsilons = [8, 12, 16, 24, 32, 48, 80]

vmiAttackM = VMIAttackModified(model=cifarResNet)
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
            correctVmiAdvWB, batch_size = computeAcc(cifarResNet, vmiAdv, targets)
            vmiCorrectWB += correctVmiAdvWB
            # Black Box Scenario
            correctVmiAdvBB, batch_size = computeAcc(cifarVt, vmiAdv, targets, is_Vt=True)
            vmiCorrectBB += correctVmiAdvBB

            total += batch_size
        vmiAccWhiteBox = 100. * vmiCorrectWB / total
        vmiAccBlackBox = 100. * vmiCorrectBB / total

        vmiWBoxRobustAccList.append(vmiAccWhiteBox)
        vmiBBoxRobustAccList.append(vmiAccBlackBox)
    print('vmiBlackBoxRobustAccList=', vmiBBoxRobustAccList)
    print('vmiWhiteBoxRobustAccList=', vmiWBoxRobustAccList)


