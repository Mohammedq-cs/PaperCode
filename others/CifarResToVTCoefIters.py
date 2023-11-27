import torch
from attacks.vmiC import VMIAttackMult
from attacks.admixC import AdmixAttackMult
from utils.datasets import get_dataset
from utils.attackUtils import filterDataSetForTwoModels
from utils.loadModels import getPreTrainedModel
from utils.others import computeAcc

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

cifarDS = get_dataset('cifar10', 'test')
loaderCifar = torch.utils.data.DataLoader(cifarDS, batch_size=128, shuffle=False, num_workers=2)

epsilons = [8, 12, 24, 32, 48]

cifarVt = getPreTrainedModel('Visual Transformer', 'cifar10')
cifarResNet = getPreTrainedModel('ResNet18', 'cifar10')

filteredDS = filterDataSetForTwoModels(cifarResNet, cifarVt, loaderCifar, False, True)
floader = torch.utils.data.DataLoader(filteredDS, batch_size=128, shuffle=False, num_workers=2)

vmiRes = []
admixRes = []
coefs = [(1, 10), (2, 10), (2, 20), (2, 40), (4, 10), (4, 20), (4, 40), (6, 10), (6, 20), (10, 40), (10, 10), (10, 20), (10, 40)]
for (coef, iters) in coefs:
    print("coef=", coef, "iters=", iters)
    admixAccList = []
    vmiAccList = []
    admixAttack = AdmixAttackMult(model=cifarResNet, image_width=32, image_height=32, momentum=1.0, num_iter=iters)
    vmiAttack = VMIAttackMult(model=cifarResNet, momentum=1.0, beta=1.5, num_iters=iters)
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

            correctAdmixAdv, batch_size = computeAcc(cifarVt, admixAdv, targets, is_Vt=True)
            correctVMIAdv, batch_size = computeAcc(cifarVt, vmiAdv, targets, is_Vt=True)

            admixCorrect += correctAdmixAdv
            vmiCorrect += correctVMIAdv
            total += batch_size

        admixAcc = admixCorrect / total
        vmiAcc = vmiCorrect / total
        if coef == 1:
            print('admixAcc=', 100. * admixAcc, 'vmiAcc=', 100. * vmiAcc)
        admixAccList.append(admixAcc)
        vmiAccList.append(vmiAcc)

    print(f'admixBBRobustAccListCoef{coef}Iters{iters}=', admixAccList)
    print(f'vmiBBRobustAccListCoef{coef}Iters{iters}=', vmiAccList)
    admixRes.append((admixAccList, f'{coef} * step size, {iters} iters'))
    vmiRes.append((vmiAccList, f'{coef} * step size, {iters} iters'))
print("vmiList=", vmiRes)
print("admixList=", admixRes)




