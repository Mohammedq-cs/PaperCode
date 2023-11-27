import torch
from attacks.vmiC import VMIAttackMult
from attacks.admixC import AdmixAttackMult
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


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

cifarDS = get_dataset('cifar10', 'test')
loaderCifar = torch.utils.data.DataLoader(cifarDS, batch_size=128, shuffle=False, num_workers=2)
cifarCNN = getPreTrainedModel('CNN', 'cifar10')
cifarPORT = getPreTrainedModel('ResNet18-PORT', 'cifar10').to(device)
exp3filteredDS = filterDataSetForTwoModels(cifarCNN, cifarPORT, loaderCifar, False, False)
floader = torch.utils.data.DataLoader(exp3filteredDS, batch_size=128, shuffle=False, num_workers=2)

epsilons = [12, 24, 32, 48]
coefs = [(1, 1), (1.75, 4), (10, 10)]
for (admixCoef, vmiCoef) in coefs:
    print("admixCoef=", admixCoef, "vmiCoef=", vmiCoef)
    admixAttack = AdmixAttackMult(model=cifarCNN, image_width=32, image_height=32, momentum=1.0)
    vmiAttack = VMIAttackMult(model=cifarCNN, momentum=1.0, beta=1.5)
    for eps in epsilons:

        if eps < 32:
            steps = torch.arange(0, ((np.ceil(eps / 255 / 0.01)) * 0.01) + 0.005, 0.005)
        else:
            steps = torch.arange(0, ((np.ceil(eps / 255 / 0.01)) * 0.01) + 0.01, 0.01)

        # black box mapped to -+eps
        admixBBoxRobustAccListEps = [0 for i in range(len(steps))]
        vmiBBoxRobustAccListEps = [0 for i in range(len(steps))]

        #  probCalc
        admixProbCntEps = [0 for i in range(len(steps))]
        vmiProbCntEps = [0 for i in range(len(steps))]
        prob_total = 0
        print("eps=", eps)
        for batch_idx, (inputs, targets) in enumerate(floader):
            inputs, targets = inputs.to(device), targets.to(device)
            # run attacks
            admixAdv = admixAttack.runAttack(inputs, targets, eps / 255, coef=admixCoef)
            vmiAdv = vmiAttack.runAttack(inputs, targets, eps / 255, coef=vmiCoef)
            maxDecAdv = batchMaxDec(inputs, eps / 255)
            # calculate Pests
            admixPest = admixAdv - inputs
            vmiPest = vmiAdv - inputs
            maxDecPest = maxDecAdv - inputs
            prob_total += inputs.numel()
            for index, threshold in enumerate(steps):
                admixFilteredEps = modify_above_threshold(admixPest, threshold, eps / 255)
                vmiFilteredEps = modify_above_threshold(vmiPest, threshold, eps / 255)

                # prob Cnt
                admixProbCntEps[index] += above_threshold_cnt(admixPest, threshold)
                vmiProbCntEps[index] += above_threshold_cnt(vmiPest, threshold)

                # new Tensors
                admixAdvFEps = torch.clamp((inputs + admixFilteredEps), 0, 1).detach().to(device)
                vmiAdvFEps = torch.clamp((inputs + vmiFilteredEps), 0, 1).detach().to(device)

                # Black Box Scenario mapped to -+eps
                correctAdmixBBFilteredAdvEps, batch_size = computeAcc(cifarPORT, admixAdvFEps, targets, is_Vt=False)
                correctVMIBBFilteredAdvEps, batch_size = computeAcc(cifarPORT, vmiAdvFEps, targets, is_Vt=False)

                admixBBoxRobustAccListEps[index] += correctAdmixBBFilteredAdvEps
                vmiBBoxRobustAccListEps[index] += correctVMIBBFilteredAdvEps

        total = len(floader.dataset)

        admixProbCountEpsT = [(100. * admixCnt / prob_total) for admixCnt in admixProbCntEps]
        vmiProbCntEpsT = [(100. * vmiCnt / prob_total) for vmiCnt in vmiProbCntEps]

        print(f'admixProbCoef{admixCoef}Eps{eps}=', admixProbCountEpsT)
        print(f'vmiProbCoef{vmiCoef}Eps{eps}=', vmiProbCntEpsT)

        # Black Box Scenario mapped to -+eps
        admixBBoxRobustAccListEpsT = [(100. * admixCorrect / total) for admixCorrect in admixBBoxRobustAccListEps]
        vmiBBoxRobustAccListEpsT = [(100. * vmiCorrect / total) for vmiCorrect in vmiBBoxRobustAccListEps]

        print(f'admixBlackBoxRobustAccListCoef{admixCoef}Eps{eps}=', admixBBoxRobustAccListEpsT)
        print(f'vmiBlackBoxRobustAccListCoef{vmiCoef}Eps{eps}=', vmiBBoxRobustAccListEpsT)