import torch
from attacks.modifiedAdmix import AdmixAttackModified
from utils.datasets import get_dataset
from utils.attackUtils import filterDataSetForTwoModels
from utils.loadModels import getPreTrainedModel
from utils.others import computeAcc

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

imagenetteDS = get_dataset('imagenette', 'test')
loaderIN = torch.utils.data.DataLoader(imagenetteDS, batch_size=40, shuffle=False, num_workers=2)

imagenetteResNet = getPreTrainedModel('ResNet18', 'imagenette')
imagenetteVt = getPreTrainedModel('Visual Transformer', 'imagenette')

filteredDS = filterDataSetForTwoModels(imagenetteResNet, imagenetteVt, loaderIN, False, True)
floader = torch.utils.data.DataLoader(filteredDS, batch_size=40, shuffle=False, num_workers=2)

epsilons = [8, 12, 16, 24, 32, 48, 80]

admixAttackM = AdmixAttackModified(model=imagenetteResNet, image_width=160, image_height=160, image_resize=180)
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
            correctAdmixAdvWB, batch_size = computeAcc(imagenetteResNet, admixAdv, targets)
            admixCorrectWB += correctAdmixAdvWB
            # Black Box Scenario
            correctAdmixAdvBB, batch_size = computeAcc(imagenetteVt, admixAdv, targets, is_Vt=True)
            admixCorrectBB += correctAdmixAdvBB

            total += batch_size
        admixAccWhiteBox = 100. * admixCorrectWB / total
        admixAccBlackBox = 100. * admixCorrectBB / total

        admixWBoxRobustAccList.append(admixAccWhiteBox)
        admixBBoxRobustAccList.append(admixAccBlackBox)
    print('admixBlackBoxRobustAccList=', admixBBoxRobustAccList)
    print('admixWhiteBoxRobustAccList=', admixWBoxRobustAccList)
