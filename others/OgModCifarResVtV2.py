import torch
from utils.datasets import get_dataset
from utils.attackUtils import filterDataSetForTwoModels
from utils.loadOGModels import getOGPreTrainedModel
from attacks.vmiOG import VMI_DI_TI_SI_FGSM
from attacks.admixOG import admix_TI_FGSM
from attacks.admix import AdmixAttack
from attacks.vmi import VMIAttack
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


def computeErrTwoModels(surrogateModel, targetModel, inputsAdvF, targetsF, surrogateIsVT=False, targetIsVT=False):
    surrogateModel.eval()
    targetModel.eval()

    outputsAdvSurrogate = surrogateModel(inputsAdvF)
    outputsAdvTarget = targetModel(inputsAdvF)

    if surrogateIsVT:
        outputsAdvSurrogate = outputsAdvSurrogate.sup
    if targetIsVT:
        outputsAdvTarget = outputsAdvTarget.sup

    _, predictedAdvSurrogate = outputsAdvSurrogate.max(1)
    _, predictedAdvTarget = outputsAdvTarget.max(1)

    true_mask = (predictedAdvSurrogate != targetsF) & (predictedAdvTarget != targetsF)
    true_cnt = true_mask.sum().item()
    batch_sizeF = (predictedAdvSurrogate != targetsF).sum().item()
    if true_cnt > batch_sizeF:
        print("should not be here")
        print(batch_sizeF.size())
        print(true_mask.size())
        print(true_cnt, batch_sizeF)

    return true_cnt, batch_sizeF


cifarDS = get_dataset('cifar10', 'test')

loaderCifar = torch.utils.data.DataLoader(cifarDS, batch_size=128, shuffle=False, num_workers=2)

cifarVtT2T = getOGPreTrainedModel('Visual Transformer', 'cifar10')
cifarResNet = getOGPreTrainedModel('ResNet18', 'cifar10')

filteredDS = filterDataSetForTwoModels(cifarResNet, cifarVtT2T, loaderCifar, False, False)
floader = torch.utils.data.DataLoader(filteredDS, batch_size=128, shuffle=False, num_workers=2)

epsilons = [4, 8, 12, 16, 24, 32, 48, 64, 80, 104, 128, 160, 192, 240]

vmiAttack = VMIAttack(model=cifarResNet, momentum=1, beta=1.5)
admixAttack = AdmixAttack(model=cifarResNet, image_width=32, image_height=32, momentum=1.0)


admixErrListMyAttack = []
admixErrListOgAttack = []
vmiErrListMyAttack = []
vmiErrListOGAttack = []

for eps in epsilons:
    admixWrongMyAttack = 0
    admixWrongOGAttack = 0
    vmiWrongMyAttack = 0
    vmiWrongOgAttack = 0

    totalAdmixMyAttack = 0
    totalAdmixOgAttack = 0
    totalVMIMyAttack = 0
    totalVMIOgAttack = 0
    print("eps=", eps)
    for batch_idx, (inputs, targets) in enumerate(floader):
        inputs, targets = inputs.to(device), targets.to(device)
        admixAdvMyAttack = admixAttack.runAttack(inputs, targets, eps / 255)

        admix_cfg = {'dataset': 'cifar', 'epsilon': eps / 255.0, 'niters': 10, 'momentum': 1.0}
        admixAdvOGAttack = admix_TI_FGSM(cifarResNet, inputs, targets, admix_cfg)

        # vmi attacks
        vmiAdvMyAttack = vmiAttack.runAttack(inputs, targets, eps / 255)
        vmi_cfg = {'dataset': 'cifar', 'epsilon': eps / 255.0, 'niters': 10, 'alpha': 1.6, 'momentum': 1.0, 'N': 20, 'beta': 1.5}
        vmiAdvOGAttack = VMI_DI_TI_SI_FGSM(cifarResNet, inputs, targets, vmi_cfg)

        wrongAdmixAdvMyAttack,  admixMyAttackBS = computeErrTwoModels(cifarResNet, cifarVtT2T, admixAdvMyAttack, targets, surrogateIsVT=False, targetIsVT=False)
        wrongAdmixAdvOgAttack,  admixOgAttackBS = computeErrTwoModels(cifarResNet, cifarVtT2T, admixAdvOGAttack, targets, surrogateIsVT=False, targetIsVT=False)
        admixWrongMyAttack += wrongAdmixAdvMyAttack
        admixWrongOGAttack += wrongAdmixAdvOgAttack
        totalAdmixMyAttack += admixMyAttackBS
        totalAdmixOgAttack += admixOgAttackBS

        wrongVMIAdvMyAttack, vmiMyAttackBS = computeErrTwoModels(cifarResNet, cifarVtT2T, vmiAdvMyAttack, targets, surrogateIsVT=False, targetIsVT=False)
        wrongVMIAdvOGAttack, vmiOgAttackBS = computeErrTwoModels(cifarResNet, cifarVtT2T, vmiAdvOGAttack, targets, surrogateIsVT=False, targetIsVT=False)
        vmiWrongMyAttack += wrongVMIAdvMyAttack
        vmiWrongOgAttack += wrongVMIAdvOGAttack
        totalVMIMyAttack += vmiMyAttackBS
        totalVMIOgAttack += vmiOgAttackBS

    admixErrMyAttack = admixWrongMyAttack / totalAdmixMyAttack
    admixErrOGAttack = admixWrongOGAttack / totalAdmixOgAttack
    print('admixErrMyAttack=', 100. * admixErrMyAttack)
    print('admixErrOGAttack=', 100. * admixErrOGAttack)
    admixErrListMyAttack.append(admixErrMyAttack)
    admixErrListOgAttack.append(admixErrOGAttack)

    vmiErrMyAttack = vmiWrongMyAttack / totalVMIMyAttack
    vmiErrOgAttack = vmiWrongOgAttack / totalVMIOgAttack
    print('vmiErrMyAttack=', 100. * vmiErrMyAttack)
    print('vmiErrOGAttack=', 100. * vmiErrOgAttack)
    vmiErrListMyAttack.append(vmiErrMyAttack)
    vmiErrListOGAttack.append(vmiErrOgAttack)

print('admixErrListMyAttack=', admixErrListMyAttack)
print('admixErrListOGAttack=', admixErrListOgAttack)
print('vmiErrListMyAttack=', vmiErrListMyAttack)
print('vmiErrListOGAttack=', vmiErrListOGAttack)


