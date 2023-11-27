import torch
from attacks.vmiOG import VMI_DI_TI_SI_FGSM
from utils.datasets import get_dataset
from utils.attackUtils import filterDataSetForTwoModels
from utils.loadModels import getPreTrainedModel
from utils.loadOGModels import getOGPreTrainedModel
from attacks.vmi import VMIAttack


def computeErrTwoModels(surrogateModel, targetModel, inputsF, inputsAdvF, targetsF, surrogateIsVT=False, targetIsVT=False):
    surrogateModel.eval()
    targetModel.eval()

    outputsAdvSurrogate = surrogateModel(inputsAdvF)
    outputsSurrogate = surrogateModel(inputsF)
    outputsAdvTarget = targetModel(inputsAdvF)
    outputsTarget = targetModel(inputsF)

    if surrogateIsVT:
        outputsAdvSurrogate = outputsAdvSurrogate.sup
        outputsSurrogate = outputsSurrogate.sup
    if targetIsVT:
        outputsAdvTarget = outputsAdvTarget.sup
        outputsTarget = outputsTarget.sup

    _, predictedAdvSurrogate = outputsAdvSurrogate.max(1)
    _, predictedSurrogate = outputsSurrogate.max(1)
    _, predictedAdvTarget = outputsAdvTarget.max(1)
    _, predictedTarget = outputsTarget.max(1)

    correct_mask = (predictedAdvSurrogate != targetsF) & (predictedAdvTarget != targetsF) & (predictedSurrogate == targetsF) & (
                predictedTarget == targetsF)
    correct = correct_mask.sum().item()
    batch_sizeF = ((predictedAdvSurrogate != targetsF) & (predictedSurrogate == targetsF) & (
                predictedTarget == targetsF)).sum().item()

    return correct, batch_sizeF


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

# og model and og attack
cifarDS = get_dataset('cifar10', 'test')
loaderCifar = torch.utils.data.DataLoader(cifarDS, batch_size=100, shuffle=False, num_workers=2)


cifarVtMy = getPreTrainedModel('Visual Transformer', 'cifar10')
cifarResNetMy = getPreTrainedModel('ResNet18', 'cifar10')

vmiErrListMy = []
epsilons = [16, 24, 32, 48]

vmiAttack = VMIAttack(model=cifarResNetMy, momentum=1.0, beta=1.5)

for eps in epsilons:
    vmiWrong = 0
    total = 0
    print("eps=", eps)
    for batch_idx, (inputs, targets) in enumerate(loaderCifar):
        inputs, targets = inputs.to(device), targets.to(device)

        vmiAdv = vmiAttack.runAttack(inputs, targets, eps/255.0)
        wrongVMIAdv, batch_size = computeErrTwoModels(cifarResNetMy, cifarVtMy, inputs, vmiAdv, targets, surrogateIsVT=False, targetIsVT=True)
        vmiWrong += wrongVMIAdv
        total += batch_size

    vmiErr = 100. * vmiWrong / total

    print('vmiErrMy', vmiErr)

    vmiErrListMy.append(vmiErr)

print('vmiErrListMy=', vmiErrListMy)

cifarVtOg = getOGPreTrainedModel('Visual Transformer', 'cifar10')
cifarResNetOg = getOGPreTrainedModel('ResNet18', 'cifar10')

'''filteredDS = filterDataSetForTwoModels(cifarResNetOg, cifarVtOg, loaderCifar, False, False)
floader = torch.utils.data.DataLoader(filteredDS, batch_size=128, shuffle=False, num_workers=2)'''


vmiErrListOg = []

for eps in epsilons:
    vmiWrong = 0
    total = 0
    print("eps=", eps)
    for batch_idx, (inputs, targets) in enumerate(loaderCifar):
        inputs, targets = inputs.to(device), targets.to(device)
        vmi_cfg = {'dataset': 'cifar10', 'epsilon': eps / 255.0, 'niters': 10, 'alpha': 1.6, 'momentum': 1.0, 'N': 20, 'beta': 1.5}
        vmiAdv = VMI_DI_TI_SI_FGSM(cifarResNetOg, inputs, targets, vmi_cfg)
        wrongVMIAdv, batch_size = computeErrTwoModels(cifarResNetOg, cifarVtOg, inputs, vmiAdv, targets, surrogateIsVT=False, targetIsVT=False)
        vmiWrong += wrongVMIAdv
        total += batch_size

    vmiErr = 100. * vmiWrong / total

    print('vmiErrOg', vmiErr)

    vmiErrListOg.append(vmiErr)

print('vmiErrListOg=', vmiErrListOg)
