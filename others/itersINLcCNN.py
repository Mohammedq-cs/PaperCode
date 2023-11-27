import torch
from attacks.vmi import VMIAttack
from attacks.admix import AdmixAttack
from utils.datasets import get_dataset
from utils.attackUtils import filterDataSetForTwoModels
from utils.loadModels import getPreTrainedModel
from utils.others import computeAcc


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

imagenetteDS = get_dataset('imagenette', 'test')
loaderIN = torch.utils.data.DataLoader(imagenetteDS, batch_size=40, shuffle=False, num_workers=2)

imagenetteLinear = getPreTrainedModel('Linear Classifier', 'imagenette')
imagenetteCNN = getPreTrainedModel('CNN', 'imagenette')

filteredDS = filterDataSetForTwoModels(imagenetteLinear, imagenetteCNN, loaderIN, False, False)
floader = torch.utils.data.DataLoader(filteredDS, batch_size=40, shuffle=False, num_workers=2)

epsilons = [8, 12, 24, 32, 48]
vmiRes = []
admixRes = []

baseIters = 10
itersMulti = [1, 2, 3, 4, 5, 6]
for itersMult in itersMulti:
    itersVal = itersMult * baseIters
    print("itersCnt=", itersVal)

    admixAccList = []
    vmiAccList = []
    admixAttack = AdmixAttack(model=imagenetteLinear, image_width=160, image_height=160, image_resize=180, momentum=1.0, num_iter=itersVal)
    vmiAttack = VMIAttack(model=imagenetteLinear, momentum=1.0, beta=1.5, num_iters=itersVal)
    for eps in epsilons:
        #print("eps=", eps)
        admixCorrect = 0
        vmiCorrect = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(floader):
            inputs, targets = inputs.to(device), targets.to(device)
            # run attacks
            admixAdv = admixAttack.runAttack(inputs, targets, eps / 255)
            vmiAdv = vmiAttack.runAttack(inputs, targets, eps / 255)

            correctAdmixAdv, batch_size = computeAcc(imagenetteCNN, admixAdv, targets, is_Vt=False)
            correctVMIAdv, batch_size = computeAcc(imagenetteCNN, vmiAdv, targets, is_Vt=False)

            admixCorrect += correctAdmixAdv
            vmiCorrect += correctVMIAdv
            total += batch_size

        admixAcc = admixCorrect / total
        vmiAcc = vmiCorrect / total
        if itersMult == 1:
            print('admixAcc=', 100. * admixAcc, 'vmiAcc=', 100. * vmiAcc)
        admixAccList.append(admixAcc)
        vmiAccList.append(vmiAcc)

    print(f'admixBBRobustAccListItersVal{itersVal}=', admixAccList)
    print(f'vmiBBRobustAccListItersVal{itersVal}=', vmiAccList)
    admixRes.append((admixAccList, f'{itersVal} iters'))
    vmiRes.append((vmiAccList, f'{itersVal} iters'))

print("vmiList=", vmiRes)
print("admixList=", admixRes)