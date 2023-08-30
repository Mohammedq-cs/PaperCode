import torch
from attacks.vmi import VMIAttack
from attacks.admix import AdmixAttack
from attacks.autoPGD import APGDAttack
from attacks.maxDecentralized import batchMaxDec
from utils.datasets import get_dataset
from utils.attackUtils import filterDataSetForTwoModels
from utils.loadModels import getPreTrainedModel
from utils.others import computeAcc

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

cifarDS = get_dataset('cifar10', 'test')
imagenetteDS = get_dataset('imagenette', 'test')

loaderCifar = torch.utils.data.DataLoader(cifarDS, batch_size=20, shuffle=False, num_workers=2)

cifarVt = getPreTrainedModel('Visual Transformer', 'cifar10')
cifarResNet = getPreTrainedModel('ResNet18', 'cifar10')

filteredDS = filterDataSetForTwoModels(cifarResNet, cifarVt, loaderCifar, False, True)
floader = torch.utils.data.DataLoader(filteredDS, batch_size=20, shuffle=False, num_workers=2)

epsilons = [4, 8, 12, 16, 24, 32, 48, 64, 80, 104, 128, 160, 192, 240]
admixAttack = AdmixAttack(model=cifarResNet, image_width=32, image_height=32)
vmiAttack = VMIAttack(model=cifarResNet)
autoPGDErrList = []
admixErrList = []
vmiErrList = []
maxDecErrList = []
maxErrList = []
for eps in epsilons:
    apgdAttack = APGDAttack(predict=cifarResNet, eps=eps / 255, device=device, is_Vt=False)
    aPGDCorrect = 0
    admixCorrect = 0
    vmiCorrect = 0
    maxDecCorrect = 0
    total = 0
    print("eps=", eps)
    for batch_idx, (inputs, targets) in enumerate(floader):
        inputs, targets = inputs.to(device), targets.to(device)
        apgdAdv = apgdAttack.perturb(inputs, targets)
        admixAdv = admixAttack.runAttack(inputs, targets, eps/255)
        vmiAdv = vmiAttack.runAttack(inputs, targets, eps/255)
        maxDecAdv = batchMaxDec(inputs, eps/255)
        correctAPGDAdv, batch_size = computeAcc(cifarVt, apgdAdv, targets, is_Vt=True)
        correctAdmixAdv, batch_size = computeAcc(cifarVt, admixAdv, targets, is_Vt=True)
        correctVMIAdv, batch_size = computeAcc(cifarVt, vmiAdv, targets, is_Vt=True)
        correctMaxDecAdv, batch_size = computeAcc(cifarVt, maxDecAdv, targets, is_Vt=True)
        aPGDCorrect += correctAPGDAdv
        admixCorrect += correctAdmixAdv
        vmiCorrect += correctVMIAdv
        maxDecCorrect += correctMaxDecAdv
        total += batch_size

    aPGDCAcc = 100.*aPGDCorrect/total
    admixAcc = 100.*admixCorrect/total
    vmiAcc = 100.*vmiCorrect/total
    maxDecAcc = 100.*maxDecCorrect/total
    maxErr = max([(100 - aPGDCAcc)/100, (100 - admixAcc)/100, (100 - vmiAcc)/100, (100 - maxDecAcc)/100])
    print('aPGDCAcc=', aPGDCAcc, ' admixAcc=', admixAcc, ' vmiAcc=', vmiAcc, ' maxDecAcc=', maxDecAcc, ' maxErr=', maxErr)
    maxErrList.append(maxErr)
    autoPGDErrList.append((100 - aPGDCAcc)/100)
    admixErrList.append((100 - admixAcc)/100)
    vmiErrList.append((100 - vmiAcc)/100)
    maxDecErrList.append((100 - maxDecAcc)/100)

print('autoPGDErrList=', autoPGDErrList)
print('admixErrList=', admixErrList)
print('vmiErrList=', vmiErrList)
print('maxErrList=', maxErrList)
print('maxDecErrList=', maxDecErrList)


