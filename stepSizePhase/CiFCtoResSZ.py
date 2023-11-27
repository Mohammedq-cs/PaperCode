import torch
from attacks.vmiC import VMIAttackC
from attacks.admixC import AdmixAttackC
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
cifarFC = getPreTrainedModel('Fully Connected', 'cifar10')
resnet18Cifar = getPreTrainedModel('ResNet18', 'cifar10')

filteredDS = filterDataSetForTwoModels(cifarFC, resnet18Cifar, loaderCifar, False, False)
floader = torch.utils.data.DataLoader(filteredDS, batch_size=128, shuffle=False, num_workers=2)
epsilons = [12, 24, 32, 48, 80]


alphas = [0.001, 0.005, 0.01, 0.05, 0.1]
for alphaVal in alphas:
    print("alpha=", alphaVal)
    admixAttack = AdmixAttackC(model=cifarFC, image_width=32, image_height=32, momentum=1.0, alpha=alphaVal)
    vmiAttack = VMIAttackC(model=cifarFC, momentum=1.0, alpha=alphaVal, beta=1.5)
    for eps in epsilons:
        apgdAttack = APGDAttack(predict=cifarFC, eps=eps / 255, device=device, is_Vt=False)

        if eps < 32:
            steps = torch.arange(0, ((np.ceil(eps / 255 / 0.01)) * 0.01) + 0.005, 0.005)
        else:
            steps = torch.arange(0, ((np.ceil(eps / 255 / 0.01)) * 0.01) + 0.01, 0.01)

        # black box mapped to -+eps
        autoPGDBBoxRobustAccListEps = [0 for i in range(len(steps))]
        admixBBoxRobustAccListEps = [0 for i in range(len(steps))]
        vmiBBoxRobustAccListEps = [0 for i in range(len(steps))]

        #  probCalc
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
                correctAPGDBBFilteredAdvEps, batch_size = computeAcc(resnet18Cifar, apgdAdvFEps, targets, is_Vt=False)
                correctAdmixBBFilteredAdvEps, batch_size = computeAcc(resnet18Cifar, admixAdvFEps, targets, is_Vt=False)
                correctVMIBBFilteredAdvEps, batch_size = computeAcc(resnet18Cifar, vmiAdvFEps, targets, is_Vt=False)

                autoPGDBBoxRobustAccListEps[index] += correctAPGDBBFilteredAdvEps
                admixBBoxRobustAccListEps[index] += correctAdmixBBFilteredAdvEps
                vmiBBoxRobustAccListEps[index] += correctVMIBBFilteredAdvEps

        total = len(floader.dataset)

        autoPGDProbCntEpsT = [(100. * aPGDCnt / prob_total) for aPGDCnt in autoPGDProbCntEps]
        admixProbCountEpsT = [(100. * admixCnt / prob_total) for admixCnt in admixProbCntEps]
        vmiProbCntEpsT = [(100. * vmiCnt / prob_total) for vmiCnt in vmiProbCntEps]

        print(f'autoPGDProbEpsMapped{alphaVal}{eps}=', autoPGDProbCntEpsT)
        print(f'admixProbEpsMapped{alphaVal}{eps}=', admixProbCountEpsT)
        print(f'vmiProbEpsMapped{alphaVal}{eps}=', vmiProbCntEpsT)

        # Black Box Scenario mapped to -+eps
        autoPGDBBoxRobustAccEpsListT = [(100. * aPGDCorrect / total) for aPGDCorrect in autoPGDBBoxRobustAccListEps]
        admixBBoxRobustAccListEpsT = [(100. * admixCorrect / total) for admixCorrect in admixBBoxRobustAccListEps]
        vmiBBoxRobustAccListEpsT = [(100. * vmiCorrect / total) for vmiCorrect in vmiBBoxRobustAccListEps]

        print(f'autoPGDBlackBoxRobustAccListEpsMapped{alphaVal}{eps}=', autoPGDBBoxRobustAccEpsListT)
        print(f'admixBlackBoxRobustAccListEpsMapped{alphaVal}{eps}=', admixBBoxRobustAccListEpsT)
        print(f'vmiBlackBoxRobustAccListEpsMapped{alphaVal}{eps}=', vmiBBoxRobustAccListEpsT)

