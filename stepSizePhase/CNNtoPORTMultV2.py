import torch
from attacks.vmiC import VMIAttackMult
from attacks.admixC import AdmixAttackMult
from utils.datasets import get_dataset
from utils.attackUtils import filterDataSetForTwoModels
from utils.loadModels import getPreTrainedModel
from utils.others import computeAcc

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

epsilons = [8, 12, 24, 32, 48]

cifarDS = get_dataset('cifar10', 'test')
loaderCifar = torch.utils.data.DataLoader(cifarDS, batch_size=128, shuffle=False, num_workers=2)
cifarCNN = getPreTrainedModel('CNN', 'cifar10')
cifarPORT = getPreTrainedModel('ResNet18-PORT', 'cifar10').to(device)
exp3filteredDS = filterDataSetForTwoModels(cifarCNN, cifarPORT, loaderCifar, False, False)
floader = torch.utils.data.DataLoader(exp3filteredDS, batch_size=128, shuffle=False, num_workers=2)


vmiRes = []
admixRes = []
coefs = [1, 1.25, 1.5, 1.75, 2, 2.25, 2.5, 2.75, 3, 3.5, 4, 6, 8, 10]
for coef in coefs:
    print("coef=", coef)
    admixAccList = []
    vmiAccList = []
    admixAttack = AdmixAttackMult(model=cifarCNN, image_width=32, image_height=32, momentum=1.0)
    vmiAttack = VMIAttackMult(model=cifarCNN, momentum=1.0, beta=1.5)
    for eps in epsilons:
        #print("eps=", eps)
        admixCorrect = 0
        vmiCorrect = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(floader):
            inputs, targets = inputs.to(device), targets.to(device)
            # run attacks
            admixAdv = admixAttack.runAttack(inputs, targets, eps / 255, coef=coef)
            vmiAdv = vmiAttack.runAttack(inputs, targets, eps / 255, coef=coef)

            correctAdmixAdv, batch_size = computeAcc(cifarPORT, admixAdv, targets, is_Vt=False)
            correctVMIAdv, batch_size = computeAcc(cifarPORT, vmiAdv, targets, is_Vt=False)

            admixCorrect += correctAdmixAdv
            vmiCorrect += correctVMIAdv
            total += batch_size

        admixAcc = admixCorrect / total
        vmiAcc = vmiCorrect / total
        if coef == 1:
            print('admixAcc=', 100. * admixAcc, 'vmiAcc=', 100. * vmiAcc)
        admixAccList.append(admixAcc)
        vmiAccList.append(vmiAcc)

    print(f'admixBBRobustAccListCoef{coef}=', admixAccList)
    print(f'vmiBBRobustAccListCoef{coef}=', vmiAccList)
    admixRes.append((admixAccList, f'{coef} * step size'))
    vmiRes.append((vmiAccList, f'{coef} * step size'))
print("vmiList=", vmiRes)
print("admixList=", admixRes)




