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

imagenetteDS = get_dataset('imagenette', 'test')
loaderIN = torch.utils.data.DataLoader(imagenetteDS, batch_size=20, shuffle=False, num_workers=2)

imagenetteResNet = getPreTrainedModel('ResNet18', 'imagenette')
imagenetteVt = getPreTrainedModel('Visual Transformer', 'imagenette')

filteredDS = filterDataSetForTwoModels(imagenetteResNet, imagenetteVt, loaderIN, False, True)
floader = torch.utils.data.DataLoader(filteredDS, batch_size=20, shuffle=False, num_workers=2)

epsilons = [4, 8, 12, 16, 24, 32, 48, 64, 80, 104, 128, 160, 192, 240]
admixAttack = AdmixAttack(model=imagenetteResNet, image_width=160, image_height=160, image_resize=180)
vmiAttack = VMIAttack(model=imagenetteResNet)
autoPGDErrList = []
admixErrList = []
vmiErrList = []
maxDecErrList = []
maxErrList = []
for eps in epsilons:
    apgdAttack = APGDAttack(predict=imagenetteResNet, eps=eps / 255, device=device, is_Vt=False)
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
        correctAPGDAdv, batch_size = computeAcc(imagenetteVt, apgdAdv, targets, is_Vt=True)
        correctAdmixAdv, batch_size = computeAcc(imagenetteVt, admixAdv, targets, is_Vt=True)
        correctVMIAdv, batch_size = computeAcc(imagenetteVt, vmiAdv, targets, is_Vt=True)
        correctMaxDecAdv, batch_size = computeAcc(imagenetteVt, maxDecAdv, targets, is_Vt=True)
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

