import torch
from utils.datasets import get_dataset
from attacks.modifiedVmi import VMIAttackModified
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
vmiAttackM = VMIAttackModified(model=imagenetteResNet)

epsilons = [12, 16, 24, 32, 48, 80]

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
            correctVmiAdvWB, batch_size = computeAcc(imagenetteResNet, vmiAdv, targets)
            vmiCorrectWB += correctVmiAdvWB
            # Black Box Scenario
            correctVmiAdvBB, batch_size = computeAcc(imagenetteVt, vmiAdv, targets, is_Vt=True)
            vmiCorrectBB += correctVmiAdvBB

            total += batch_size
        vmiAccWhiteBox = 100. * vmiCorrectWB / total
        vmiAccBlackBox = 100. * vmiCorrectBB / total

        vmiWBoxRobustAccList.append(vmiAccWhiteBox)
        vmiBBoxRobustAccList.append(vmiAccBlackBox)
    print('vmiBlackBoxRobustAccList=', vmiBBoxRobustAccList)
    print('vmiWhiteBoxRobustAccList=', vmiWBoxRobustAccList)


