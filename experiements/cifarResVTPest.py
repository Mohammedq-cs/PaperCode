import torch
from attacks.admix import AdmixAttack
from utils.datasets import get_dataset
from utils.attackUtils import filterDataSetForTwoModels
from utils.loadModels import getPreTrainedModel
from utils.others import computeAcc

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

cifarDS = get_dataset('cifar10', 'test')
loaderCifar = torch.utils.data.DataLoader(cifarDS, batch_size=80, shuffle=False, num_workers=2)
cifarVt = getPreTrainedModel('Visual Transformer', 'cifar10')
cifarResNet = getPreTrainedModel('ResNet18', 'cifar10')

filteredDS = filterDataSetForTwoModels(cifarResNet, cifarVt, loaderCifar, False, True)
floader = torch.utils.data.DataLoader(filteredDS, batch_size=40, shuffle=False, num_workers=2)

epsilons = [4, 8, 12, 16, 24, 32, 48, 64, 80, 104, 128, 160, 192, 240]
admixAttack = AdmixAttack(model=cifarResNet, image_width=32, image_height=32)
admixErrList = []
pestsList = []
for eps in epsilons:
    admixCorrect = 0
    total = 0
    pests = []
    print("eps=", eps)
    for batch_idx, (inputs, targets) in enumerate(floader):
        inputs, targets = inputs.to(device), targets.to(device)
        admixAdv = admixAttack.runAttack(inputs, targets, eps / 255)
        correctAdmixAdv, batch_size = computeAcc(cifarVt, admixAdv, targets, is_Vt=True)
        pest = admixAdv - inputs
        flattened_list = pest.view(-1).tolist()
        pests.extend(flattened_list)
        admixCorrect += correctAdmixAdv
        total += batch_size
    admixAcc = 100. * admixCorrect / total
    print('admixAcc=', admixAcc)
    admixErrList.append((100 - admixAcc) / 100)
    pestsList.append(pests)
    print(len(pests))
print('admixErrList=', admixErrList)
print()

with open('admixPost.txt', 'w') as f:
    lines = [' '.join(map(str, sublist)) + '\n' for sublist in pestsList]
    f.writelines(lines)

