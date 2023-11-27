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

imagenetteDS = get_dataset('imagenette', 'test')
loaderIN = torch.utils.data.DataLoader(imagenetteDS, batch_size=40, shuffle=False, num_workers=2)

imagenetteResNet = getPreTrainedModel('ResNet18', 'imagenette')
imagenetteVt = getPreTrainedModel('Visual Transformer', 'imagenette')

filteredDS = filterDataSetForTwoModels(imagenetteResNet, imagenetteVt, loaderIN, False, True)
floader = torch.utils.data.DataLoader(filteredDS, batch_size=40, shuffle=False, num_workers=2)

epsilons = [12, 16, 24, 32, 48, 80]
alphas = [0.01, 0.1, 0.001]
for alphaVal in alphas:
    print("alpha=", alphaVal)
    admixAttack = AdmixAttackC(model=imagenetteResNet, image_width=160, image_height=160, image_resize=180, momentum=1.0, alpha=alphaVal)
    vmiAttack = VMIAttackC(model=imagenetteResNet, momentum=1.0, alpha=alphaVal, beta=1.5)

    for eps in epsilons:
        if eps < 32:
            steps = torch.arange(0, ((np.ceil(eps / 255 / 0.01)) * 0.01) + 0.005, 0.005)
        else:
            steps = torch.arange(0, ((np.ceil(eps / 255 / 0.01)) * 0.01) + 0.01, 0.01)

        apgdAttack = APGDAttack(predict=imagenetteResNet, eps=eps / 255, device=device, is_Vt=False)
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
            prob_total += inputs.numel()
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

                # prob Cnt
                autoPGDProbCntEps[index] += above_threshold_cnt(apgdPest, threshold)
                admixProbCntEps[index] += above_threshold_cnt(admixPest, threshold)
                vmiProbCntEps[index] += above_threshold_cnt(vmiPest, threshold)

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

        total = len(floader.dataset)
        # prob calc
        autoPGDProbCntEpsT = [(100. * aPGDCnt / prob_total) for aPGDCnt in autoPGDProbCntEps]
        admixProbCntEpsT = [(100. * admixCnt / prob_total) for admixCnt in admixProbCntEps]
        vmiProbCntEpsT = [(100. * vmiCnt / prob_total) for vmiCnt in vmiProbCntEps]

        print(f'autoPGDProbEpsMapped{alphaVal}{eps}=', autoPGDProbCntEpsT)
        print(f'admixProbEpsMapped{alphaVal}{eps}=', admixProbCntEpsT)
        print(f'vmiProbEpsMapped{alphaVal}{eps}=', vmiProbCntEpsT)

        # Black Box Scenario mapped to -+eps
        autoPGDBBoxRobustAccEpsListT = [(100. * aPGDCorrect / total) for aPGDCorrect in autoPGDBBoxRobustAccListEps]
        admixBBoxRobustAccListEpsT = [(100. * admixCorrect / total) for admixCorrect in admixBBoxRobustAccListEps]
        vmiBBoxRobustAccListEpsT = [(100. * vmiCorrect / total) for vmiCorrect in vmiBBoxRobustAccListEps]

        print(f'autoPGDBlackBoxRobustAccListEpsMapped{alphaVal}{eps}=', autoPGDBBoxRobustAccEpsListT)
        print(f'admixBlackBoxRobustAccListEpsMapped{alphaVal}{eps}=', admixBBoxRobustAccListEpsT)
        print(f'vmiBlackBoxRobustAccListEpsMapped{alphaVal}{eps}=', vmiBBoxRobustAccListEpsT)

