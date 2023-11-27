import torch
from attacks.vmi import VMIAttack
from attacks.admix import AdmixAttack
from attacks.autoPGD import APGDAttack
from attacks.maxDecentralized import batchMaxDec
from utils.datasets import get_dataset
from utils.attackUtils import filterDataSetForTwoModels
from utils.loadModels import getPreTrainedModel
from utils.others import computeAcc
import numpy as np


def modify_below_threshold(input_tensor: torch.tensor, thresholdValue: float) -> torch.tensor:
    condition = (thresholdValue >= input_tensor) & (input_tensor >= -thresholdValue)
    output_tensor = torch.where(condition, 0.0, input_tensor)
    return output_tensor


def modify_above_threshold(input_tensor: torch.tensor, thresholdValue: float, epsValue: float) -> torch.tensor:
    condition1 = (thresholdValue <= input_tensor)
    output_tensor = torch.where(condition1, epsValue, input_tensor)
    condition2 = (input_tensor <= -thresholdValue)
    output_tensor2 = torch.where(condition2, -epsValue, output_tensor)
    return output_tensor2

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

imagenetteDS = get_dataset('imagenette', 'test')
loaderIN = torch.utils.data.DataLoader(imagenetteDS, batch_size=40, shuffle=False, num_workers=2)

imagenetteResNet = getPreTrainedModel('ResNet18', 'imagenette')
imagenetteVt = getPreTrainedModel('Visual Transformer', 'imagenette')

filteredDS = filterDataSetForTwoModels(imagenetteResNet, imagenetteVt, loaderIN, False, True)
floader = torch.utils.data.DataLoader(filteredDS, batch_size=40, shuffle=False, num_workers=2)

epsilons = [8, 12, 16, 24, 32, 48, 80]
wantedEpsilons = [12, 24, 32, 48]

admixAttack = AdmixAttack(model=imagenetteResNet, image_width=160, image_height=160, image_resize=180)
vmiAttack = VMIAttack(model=imagenetteResNet)

for eps in epsilons:
    if eps < 32:
        steps = torch.arange(0, ((np.ceil(eps / 255 / 0.01)) * 0.01) + 0.005, 0.005)
    else:
        steps = torch.arange(0, ((np.ceil(eps / 255 / 0.01)) * 0.01) + 0.01, 0.01)

    apgdAttack = APGDAttack(predict=imagenetteResNet, eps=eps / 255, device=device, is_Vt=False)
    # black box mapped to zero
    autoPGDBBoxRobustAccListZero = [0 for i in range(len(steps))]
    admixBBoxRobustAccListZero = [0 for i in range(len(steps))]
    vmiBBoxRobustAccListZero = [0 for i in range(len(steps))]
    # white box mapped to zero
    autoPGDWBoxRobustAccListZero = [0 for i in range(len(steps))]
    admixWBoxRobustAccListZero = [0 for i in range(len(steps))]
    vmiWBoxRobustAccListZero = [0 for i in range(len(steps))]

    # black box mapped to -+eps
    autoPGDBBoxRobustAccListEps = [0 for i in range(len(steps))]
    admixBBoxRobustAccListEps = [0 for i in range(len(steps))]
    vmiBBoxRobustAccListEps = [0 for i in range(len(steps))]
    # white box mapped to -+eps
    autoPGDWBoxRobustAccListEps = [0 for i in range(len(steps))]
    admixWBoxRobustAccListEps = [0 for i in range(len(steps))]
    vmiWBoxRobustAccListEps = [0 for i in range(len(steps))]

    print("eps=", eps)
    for batch_idx, (inputs, targets) in enumerate(floader):
        inputs, targets = inputs.to(device), targets.to(device)
        # run attacks
        apgdAdv = apgdAttack.perturb(inputs, targets)
        admixAdv = admixAttack.runAttack(inputs, targets, eps / 255)
        vmiAdv = vmiAttack.runAttack(inputs, targets, eps / 255)
        maxDecAdv = batchMaxDec(inputs, eps / 255)
        # calculate Pests
        admixPest = admixAdv - inputs
        apgdPest = apgdAdv - inputs
        vmiPest = vmiAdv - inputs
        maxDecPest = maxDecAdv - inputs
        for index, threshold in enumerate(steps):
            admixFilteredEps = modify_above_threshold(admixPest, threshold, eps / 255)
            vmiFilteredEps = modify_above_threshold(vmiPest, threshold, eps / 255)
            apgdFilteredEps = modify_above_threshold(apgdPest, threshold, eps / 255)

            # new Tensors
            apgdAdvFEps = torch.clamp((inputs + apgdFilteredEps), 0, 1).detach().to(device)
            admixAdvFEps = torch.clamp((inputs + admixFilteredEps), 0, 1).detach().to(device)
            vmiAdvFEps = torch.clamp((inputs + vmiFilteredEps), 0, 1).detach().to(device)

            # Black Box Scenario mapped to -+eps
            correctAPGDBBFilteredAdvEps, batch_size = computeAcc(imagenetteVt, apgdAdvFEps, targets, is_Vt=True)
            correctAdmixBBFilteredAdvEps, batch_size = computeAcc(imagenetteVt, admixAdvFEps, targets, is_Vt=True)
            correctVMIBBFilteredAdvEps, batch_size = computeAcc(imagenetteVt, vmiAdvFEps, targets, is_Vt=True)

            autoPGDBBoxRobustAccListEps[index] += correctAPGDBBFilteredAdvEps
            admixBBoxRobustAccListEps[index] += correctAdmixBBFilteredAdvEps
            vmiBBoxRobustAccListEps[index] += correctVMIBBFilteredAdvEps

            # White Box Scenario mapped to -+eps
            correctAPGDWBFilteredAdvEps, batch_size = computeAcc(imagenetteResNet, apgdAdvFEps, targets, is_Vt=False)
            correctAdmixWBFilteredAdvEps, batch_size = computeAcc(imagenetteResNet, admixAdvFEps, targets, is_Vt=False)
            correctVMIWBFilteredAdvEps, batch_size = computeAcc(imagenetteResNet, vmiAdvFEps, targets, is_Vt=False)

            autoPGDWBoxRobustAccListEps[index] += correctAPGDWBFilteredAdvEps
            admixWBoxRobustAccListEps[index] += correctAdmixWBFilteredAdvEps
            vmiWBoxRobustAccListEps[index] += correctVMIWBFilteredAdvEps

            admixFiltered = modify_below_threshold(admixPest, threshold)
            vmiFiltered = modify_below_threshold(vmiPest, threshold)
            apgdFiltered = modify_below_threshold(apgdPest, threshold)

            # new Tensors mapped to zero
            apgdAdvF = (inputs + apgdFiltered).to(device)
            admixAdvF = (inputs + admixFiltered).to(device)
            vmiAdvF = (inputs + vmiFiltered).to(device)

            # Black Box Scenario mapped to zero
            correctAPGDBBFilteredAdv, batch_size = computeAcc(imagenetteVt, apgdAdvF, targets, is_Vt=True)
            correctAdmixBBFilteredAdv, batch_size = computeAcc(imagenetteVt, admixAdvF, targets, is_Vt=True)
            correctVMIBBFilteredAdv, batch_size = computeAcc(imagenetteVt, vmiAdvF, targets, is_Vt=True)

            autoPGDBBoxRobustAccListZero[index] += correctAPGDBBFilteredAdv
            admixBBoxRobustAccListZero[index] += correctAdmixBBFilteredAdv
            vmiBBoxRobustAccListZero[index] += correctVMIBBFilteredAdv

            # White Box Scenario mapped to zero
            correctAPGDWBFilteredAdv, batch_size = computeAcc(imagenetteResNet, apgdAdvF, targets, is_Vt=False)
            correctAdmixWBFilteredAdv, batch_size = computeAcc(imagenetteResNet, admixAdvF, targets, is_Vt=False)
            correctVMIWBFilteredAdv, batch_size = computeAcc(imagenetteResNet, vmiAdvF, targets, is_Vt=False)

            autoPGDWBoxRobustAccListZero[index] += correctAPGDWBFilteredAdv
            admixWBoxRobustAccListZero[index] += correctAdmixWBFilteredAdv
            vmiWBoxRobustAccListZero[index] += correctVMIWBFilteredAdv

    total = len(floader.dataset)
    # Black Box Scenario mapped to zero
    autoPGDBBoxRobustAccZeroListT = [(100. * aPGDCorrect / total) for aPGDCorrect in autoPGDBBoxRobustAccListZero]
    admixBBoxRobustAccListZeroT = [(100. * admixCorrect / total) for admixCorrect in admixBBoxRobustAccListZero]
    vmiBBoxRobustAccListZeroT = [(100. * vmiCorrect / total) for vmiCorrect in vmiBBoxRobustAccListZero]

    print('autoPGDBlackBoxRobustAccListZeroMapped=', autoPGDBBoxRobustAccZeroListT)
    print('admixBlackBoxRobustAccListZeroMapped=', admixBBoxRobustAccListZeroT)
    print('vmiBlackBoxRobustAccListZeroMapped=', vmiBBoxRobustAccListZeroT)

    # White Box Scenario mapped to zero
    autoPGDWBoxRobustAccListZeroT = [(100. * aPGDCorrect / total) for aPGDCorrect in autoPGDWBoxRobustAccListZero]
    admixWBoxRobustAccListZeroT = [(100. * admixCorrect / total) for admixCorrect in admixWBoxRobustAccListZero]
    vmiWBoxRobustAccListZeroT = [(100. * vmiCorrect / total) for vmiCorrect in vmiWBoxRobustAccListZero]

    print('autoPGDWhiteBoxRobustAccListZeroMapped=', autoPGDWBoxRobustAccListZeroT)
    print('admixWhiteBoxRobustAccListZeroMapped=', admixWBoxRobustAccListZeroT)
    print('vmiWhiteBoxRobustAccListZeroMapped=', vmiWBoxRobustAccListZeroT)

    # Black Box Scenario mapped to -+eps
    autoPGDBBoxRobustAccEpsListT = [(100. * aPGDCorrect / total) for aPGDCorrect in autoPGDBBoxRobustAccListEps]
    admixBBoxRobustAccListEpsT = [(100. * admixCorrect / total) for admixCorrect in admixBBoxRobustAccListEps]
    vmiBBoxRobustAccListEpsT = [(100. * vmiCorrect / total) for vmiCorrect in vmiBBoxRobustAccListEps]

    print('autoPGDBlackBoxRobustAccListEpsMapped=', autoPGDBBoxRobustAccEpsListT)
    print('admixBlackBoxRobustAccListEpsMapped=', admixBBoxRobustAccListEpsT)
    print('vmiBlackBoxRobustAccListEpsMapped=', vmiBBoxRobustAccListEpsT)

    # White Box Scenario mapped to -+eps
    autoPGDWBoxRobustAccListEpsT = [(100. * aPGDCorrect / total) for aPGDCorrect in autoPGDWBoxRobustAccListEps]
    admixWBoxRobustAccListEpsT = [(100. * admixCorrect / total) for admixCorrect in admixWBoxRobustAccListEps]
    vmiWBoxRobustAccListEpsT = [(100. * vmiCorrect / total) for vmiCorrect in vmiWBoxRobustAccListEps]

    print('autoPGDWhiteBoxRobustAccListEpsMapped=', autoPGDWBoxRobustAccListEpsT)
    print('admixWhiteBoxRobustAccListEpsMapped=', admixWBoxRobustAccListEpsT)
    print('vmiWhiteBoxRobustAccListEpsMapped=', vmiWBoxRobustAccListEpsT)