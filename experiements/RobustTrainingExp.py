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

epsilons = [4, 8, 12, 16, 24, 32, 48, 64, 80, 104, 128, 160, 192, 240]
cifarDS = get_dataset('cifar10', 'test')
loaderCifar = torch.utils.data.DataLoader(cifarDS, batch_size=20, shuffle=False, num_workers=2)
cifarCNN = getPreTrainedModel('CNN', 'cifar10')

print("starting experiment 1 CNN ->  ResNet18")
resnet18Cifar = getPreTrainedModel('ResNet18', 'cifar10')

exp1filteredDS = filterDataSetForTwoModels(cifarCNN, resnet18Cifar, loaderCifar, False, False)
exp1floader = torch.utils.data.DataLoader(exp1filteredDS, batch_size=20, shuffle=False, num_workers=2)

admixAttack = AdmixAttack(model=cifarCNN, image_width=32, image_height=32)
vmiAttack = VMIAttack(model=cifarCNN)
autoPGDErrList = []
admixErrList = []
vmiErrList = []
maxDecErrList = []
maxErrList = []
for eps in epsilons:
    apgdAttack = APGDAttack(predict=cifarCNN, eps=eps / 255, device=device, is_Vt=False)
    aPGDCorrect = 0
    admixCorrect = 0
    vmiCorrect = 0
    maxDecCorrect = 0
    total = 0
    print("eps=", eps)
    for batch_idx, (inputs, targets) in enumerate(exp1floader):
        inputs, targets = inputs.to(device), targets.to(device)
        apgdAdv = apgdAttack.perturb(inputs, targets)
        admixAdv = admixAttack.runAttack(inputs, targets, eps/255)
        vmiAdv = vmiAttack.runAttack(inputs, targets, eps/255)
        maxDecAdv = batchMaxDec(inputs, eps/255)
        correctAPGDAdv, batch_size = computeAcc(resnet18Cifar, apgdAdv, targets, is_Vt=False)
        correctAdmixAdv, batch_size = computeAcc(resnet18Cifar, admixAdv, targets, is_Vt=False)
        correctVMIAdv, batch_size = computeAcc(resnet18Cifar, vmiAdv, targets, is_Vt=False)
        correctMaxDecAdv, batch_size = computeAcc(resnet18Cifar, maxDecAdv, targets, is_Vt=False)
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

print("starting experiment 2 CNN ->  ResNet20-TRSEnsemble")
cifarTrs = getPreTrainedModel('ResNet20-TRSEnsemble', 'cifar10')
exp2filteredDS = filterDataSetForTwoModels(cifarCNN, cifarTrs, loaderCifar, False, False)
exp2floader = torch.utils.data.DataLoader(exp2filteredDS, batch_size=20, shuffle=False, num_workers=2)

admixAttack = AdmixAttack(model=cifarCNN, image_width=32, image_height=32)
vmiAttack = VMIAttack(model=cifarCNN)
autoPGDErrList = []
admixErrList = []
vmiErrList = []
maxDecErrList = []
maxErrList = []
for eps in epsilons:
    apgdAttack = APGDAttack(predict=cifarCNN, eps=eps / 255, device=device, is_Vt=False)
    aPGDCorrect = 0
    admixCorrect = 0
    vmiCorrect = 0
    maxDecCorrect = 0
    total = 0
    print("eps=", eps)
    for batch_idx, (inputs, targets) in enumerate(exp2floader):
        inputs, targets = inputs.to(device), targets.to(device)
        apgdAdv = apgdAttack.perturb(inputs, targets)
        admixAdv = admixAttack.runAttack(inputs, targets, eps/255)
        vmiAdv = vmiAttack.runAttack(inputs, targets, eps/255)
        maxDecAdv = batchMaxDec(inputs, eps/255)
        correctAPGDAdv, batch_size = computeAcc(cifarTrs, apgdAdv, targets, is_Vt=False)
        correctAdmixAdv, batch_size = computeAcc(cifarTrs, admixAdv, targets, is_Vt=False)
        correctVMIAdv, batch_size = computeAcc(cifarTrs, vmiAdv, targets, is_Vt=False)
        correctMaxDecAdv, batch_size = computeAcc(cifarTrs, maxDecAdv, targets, is_Vt=False)
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

print("starting experiment 3 CNN ->  ResNet18PORT")
cifarPORT = getPreTrainedModel('ResNet18-PORT', 'cifar10')
exp3filteredDS = filterDataSetForTwoModels(cifarCNN, cifarPORT, loaderCifar, False, False)
exp3floader = torch.utils.data.DataLoader(exp3filteredDS, batch_size=20, shuffle=False, num_workers=2)

admixAttack = AdmixAttack(model=cifarCNN, image_width=32, image_height=32)
vmiAttack = VMIAttack(model=cifarCNN)
autoPGDErrList = []
admixErrList = []
vmiErrList = []
maxDecErrList = []
maxErrList = []
for eps in epsilons:
    apgdAttack = APGDAttack(predict=cifarCNN, eps=eps / 255, device=device, is_Vt=False)
    aPGDCorrect = 0
    admixCorrect = 0
    vmiCorrect = 0
    maxDecCorrect = 0
    total = 0
    print("eps=", eps)
    for batch_idx, (inputs, targets) in enumerate(exp3floader):
        inputs, targets = inputs.to(device), targets.to(device)
        apgdAdv = apgdAttack.perturb(inputs, targets)
        admixAdv = admixAttack.runAttack(inputs, targets, eps/255)
        vmiAdv = vmiAttack.runAttack(inputs, targets, eps/255)
        maxDecAdv = batchMaxDec(inputs, eps/255)
        correctAPGDAdv, batch_size = computeAcc(cifarPORT, apgdAdv, targets, is_Vt=False)
        correctAdmixAdv, batch_size = computeAcc(cifarPORT, admixAdv, targets, is_Vt=False)
        correctVMIAdv, batch_size = computeAcc(cifarPORT, vmiAdv, targets, is_Vt=False)
        correctMaxDecAdv, batch_size = computeAcc(cifarPORT, maxDecAdv, targets, is_Vt=False)
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
