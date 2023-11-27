import torch
from utils.datasets import get_dataset
from attacks.modifiedAdmix import AdmixAttackModified
from utils.attackUtils import filterDataSetForTwoModels
from utils.loadModels import getPreTrainedModel
from utils.others import computeAcc
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

cifarDS = get_dataset('cifar10', 'test')
loaderCifar = torch.utils.data.DataLoader(cifarDS, batch_size=64, shuffle=False, num_workers=2)

cifarVt = getPreTrainedModel('Visual Transformer', 'cifar10')
cifarResNet = getPreTrainedModel('ResNet18', 'cifar10')

filteredDS = filterDataSetForTwoModels(cifarResNet, cifarVt, loaderCifar, False, True)
floader = torch.utils.data.DataLoader(filteredDS, batch_size=64, shuffle=False, num_workers=2)

epsilons = [8, 12, 16, 24, 32, 48, 80]

admixAttackM = AdmixAttackModified(model=cifarResNet, image_width=32, image_height=32)
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
            correctAdmixAdvWB, batch_size = computeAcc(cifarResNet, admixAdv, targets)
            admixCorrectWB += correctAdmixAdvWB
            # Black Box Scenario
            correctAdmixAdvBB, batch_size = computeAcc(cifarVt, admixAdv, targets, is_Vt=True)
            admixCorrectBB += correctAdmixAdvBB

            total += batch_size
        admixAccWhiteBox = 100. * admixCorrectWB / total
        admixAccBlackBox = 100. * admixCorrectBB / total

        admixWBoxRobustAccList.append(admixAccWhiteBox)
        admixBBoxRobustAccList.append(admixAccBlackBox)
    print('admixBlackBoxRobustAccList=', admixBBoxRobustAccList)
    print('admixWhiteBoxRobustAccList=', admixWBoxRobustAccList)


