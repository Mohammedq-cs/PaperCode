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

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


def modify_below_threshold(input_tensor: torch.tensor, thresholdValue: float) -> torch.tensor:
    condition = (thresholdValue >= input_tensor) & (input_tensor >= -thresholdValue)
    output_tensor = torch.where(condition, 0.0, input_tensor)
    return output_tensor


def below_threshold_cnt(input_tensor: torch.tensor, thresholdValue: float) -> torch.tensor:
    condition = (thresholdValue >= input_tensor) & (input_tensor >= -thresholdValue)
    element_count = torch.sum(condition).item()
    return element_count


def modify_above_threshold(input_tensor: torch.tensor, thresholdValue: float, epsValue: float) -> torch.tensor:
    condition1 = (thresholdValue <= input_tensor)
    output_tensor = torch.where(condition1, epsValue, input_tensor)
    condition2 = (input_tensor <= -thresholdValue)
    output_tensor2 = torch.where(condition2, -epsValue, output_tensor)
    return output_tensor2


def above_threshold_cnt(input_tensor: torch.tensor, thresholdValue: float) -> torch.tensor:
    condition = (thresholdValue <= input_tensor) | (input_tensor <= -thresholdValue)
    element_count = torch.sum(condition).item()
    return element_count


cifarDS = get_dataset('cifar10', 'test')
loaderCifar = torch.utils.data.DataLoader(cifarDS, batch_size=128, shuffle=False, num_workers=2)
cifarCNN = getPreTrainedModel('CNN', 'cifar10')
cifarPORT = getPreTrainedModel('ResNet18-PORT', 'cifar10').to(device)
exp3filteredDS = filterDataSetForTwoModels(cifarCNN, cifarPORT, loaderCifar, False, False)
floader = torch.utils.data.DataLoader(exp3filteredDS, batch_size=128, shuffle=False, num_workers=2)

epsilons = [12, 24, 32, 48, 80]

admixAttack = AdmixAttack(model=cifarCNN, image_width=32, image_height=32)
vmiAttack = VMIAttack(model=cifarCNN, beta=1.5)
print("starting experiment")

for eps in epsilons:
    apgdAttack = APGDAttack(predict=cifarCNN, eps=eps / 255, device=device, is_Vt=False)

    if eps < 32:
        steps = torch.arange(0, ((np.ceil(eps / 255 / 0.01)) * 0.01) + 0.005, 0.005)
    else:
        steps = torch.arange(0, ((np.ceil(eps / 255 / 0.01)) * 0.01) + 0.01, 0.01)

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

    #  probCalc
    autoPGDProbCntZero = [0 for i in range(len(steps))]
    admixProbCntZero = [0 for i in range(len(steps))]
    vmiProbCntZero = [0 for i in range(len(steps))]

    autoPGDProbCntEps = [0 for i in range(len(steps))]
    admixProbCntEps = [0 for i in range(len(steps))]
    vmiProbCntEps = [0 for i in range(len(steps))]
    prob_total = 0
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
        prob_total += inputs.numel()
        for index, threshold in enumerate(steps):
            admixFilteredEps = modify_above_threshold(admixPest, threshold, eps / 255)
            vmiFilteredEps = modify_above_threshold(vmiPest, threshold, eps / 255)
            apgdFilteredEps = modify_above_threshold(apgdPest, threshold, eps / 255)

            # prob Cnt
            autoPGDProbCntEps[index] += above_threshold_cnt(apgdPest, threshold)
            admixProbCntEps[index] += above_threshold_cnt(admixPest, threshold)
            vmiProbCntEps[index] += above_threshold_cnt(vmiPest, threshold)

            # new Tensors
            apgdAdvFEps = torch.clamp((inputs + apgdFilteredEps), 0, 1).detach().to(device)
            admixAdvFEps = torch.clamp((inputs + admixFilteredEps), 0, 1).detach().to(device)
            vmiAdvFEps = torch.clamp((inputs + vmiFilteredEps), 0, 1).detach().to(device)

            # Black Box Scenario mapped to -+eps
            correctAPGDBBFilteredAdvEps, batch_size = computeAcc(cifarPORT, apgdAdvFEps, targets, is_Vt=False)
            correctAdmixBBFilteredAdvEps, batch_size = computeAcc(cifarPORT, admixAdvFEps, targets, is_Vt=False)
            correctVMIBBFilteredAdvEps, batch_size = computeAcc(cifarPORT, vmiAdvFEps, targets, is_Vt=False)

            autoPGDBBoxRobustAccListEps[index] += correctAPGDBBFilteredAdvEps
            admixBBoxRobustAccListEps[index] += correctAdmixBBFilteredAdvEps
            vmiBBoxRobustAccListEps[index] += correctVMIBBFilteredAdvEps

            # White Box Scenario mapped to -+eps
            correctAPGDWBFilteredAdvEps, batch_size = computeAcc(cifarCNN, apgdAdvFEps, targets, is_Vt=False)
            correctAdmixWBFilteredAdvEps, batch_size = computeAcc(cifarCNN, admixAdvFEps, targets, is_Vt=False)
            correctVMIWBFilteredAdvEps, batch_size = computeAcc(cifarCNN, vmiAdvFEps, targets, is_Vt=False)

            autoPGDWBoxRobustAccListEps[index] += correctAPGDWBFilteredAdvEps
            admixWBoxRobustAccListEps[index] += correctAdmixWBFilteredAdvEps
            vmiWBoxRobustAccListEps[index] += correctVMIWBFilteredAdvEps

            admixFiltered = modify_below_threshold(admixPest, threshold)
            vmiFiltered = modify_below_threshold(vmiPest, threshold)
            apgdFiltered = modify_below_threshold(apgdPest, threshold)

            autoPGDProbCntZero[index] += below_threshold_cnt(apgdPest, threshold)
            admixProbCntZero[index] += below_threshold_cnt(admixPest, threshold)
            vmiProbCntZero[index] += below_threshold_cnt(vmiPest, threshold)

            # new Tensors mapped to zero
            apgdAdvF = (inputs + apgdFiltered).to(device)
            admixAdvF = (inputs + admixFiltered).to(device)
            vmiAdvF = (inputs + vmiFiltered).to(device)

            # Black Box Scenario mapped to zero
            correctAPGDBBFilteredAdv, batch_size = computeAcc(cifarPORT, apgdAdvF, targets, is_Vt=False)
            correctAdmixBBFilteredAdv, batch_size = computeAcc(cifarPORT, admixAdvF, targets, is_Vt=False)
            correctVMIBBFilteredAdv, batch_size = computeAcc(cifarPORT, vmiAdvF, targets, is_Vt=False)

            autoPGDBBoxRobustAccListZero[index] += correctAPGDBBFilteredAdv
            admixBBoxRobustAccListZero[index] += correctAdmixBBFilteredAdv
            vmiBBoxRobustAccListZero[index] += correctVMIBBFilteredAdv

            # White Box Scenario mapped to zero
            correctAPGDWBFilteredAdv, batch_size = computeAcc(cifarCNN, apgdAdvF, targets, is_Vt=False)
            correctAdmixWBFilteredAdv, batch_size = computeAcc(cifarCNN, admixAdvF, targets, is_Vt=False)
            correctVMIWBFilteredAdv, batch_size = computeAcc(cifarCNN, vmiAdvF, targets, is_Vt=False)

            autoPGDWBoxRobustAccListZero[index] += correctAPGDWBFilteredAdv
            admixWBoxRobustAccListZero[index] += correctAdmixWBFilteredAdv
            vmiWBoxRobustAccListZero[index] += correctVMIWBFilteredAdv

    total = len(floader.dataset)

    autoPGDProbCntZeroT = [(100. * aPGDCnt / prob_total) for aPGDCnt in autoPGDProbCntZero]
    admixProbCountZeroT = [(100. * admixCnt / prob_total) for admixCnt in admixProbCntZero]
    vmiProbCntZeroT = [(100. * vmiCnt / prob_total) for vmiCnt in vmiProbCntZero]

    autoPGDProbCntEpsT = [(100. * aPGDCnt / prob_total) for aPGDCnt in autoPGDProbCntEps]
    admixProbCountEpsT = [(100. * admixCnt / prob_total) for admixCnt in admixProbCntEps]
    vmiProbCntEpsT = [(100. * vmiCnt / prob_total) for vmiCnt in vmiProbCntEps]

    print('autoPGDProbZeroMapped=', autoPGDProbCntZeroT)
    print('admixProbZeroMapped=', admixProbCountZeroT)
    print('vmiProbZeroMapped=', vmiProbCntZeroT)

    print('autoPGDProbEpsMapped=', autoPGDProbCntEpsT)
    print('admixProbEpsMapped=', admixProbCountEpsT)
    print('vmiProbEpsMapped=', vmiProbCntEpsT)

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



