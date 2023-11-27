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


def modify_above_threshold(input_tensor: torch.tensor, thresholdValue: float, epsValue: float) -> torch.tensor:
    condition1 = (thresholdValue < input_tensor)
    output_tensor = torch.where(condition1, epsValue, input_tensor)
    condition2 = (input_tensor < -thresholdValue)
    output_tensor2 = torch.where(condition2, -epsValue, output_tensor)
    return output_tensor2


def calculate_pmf_and_bin_edges(eps, values, bin_width=0.01):
    min_value = np.floor(-eps / 255 / bin_width) * bin_width
    max_value = np.ceil(eps / 255 / bin_width) * bin_width
    bin_edges = np.arange(min_value, max_value + bin_width, bin_width)

    counts, _ = np.histogram(values, bins=bin_edges, density=True)
    probs = counts / float(counts.sum())

    return probs, bin_edges[:-1]


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

cifarDS = get_dataset('cifar10', 'test')
loaderCifar = torch.utils.data.DataLoader(cifarDS, batch_size=128, shuffle=False, num_workers=2)

cifarVt = getPreTrainedModel('Visual Transformer', 'cifar10')
cifarResNet = getPreTrainedModel('ResNet18', 'cifar10')

filteredDS = filterDataSetForTwoModels(cifarResNet, cifarVt, loaderCifar, False, True)
floader = torch.utils.data.DataLoader(filteredDS, batch_size=128, shuffle=False, num_workers=2)

epsilons = [12, 16, 24, 32, 48, 80]
wantedEpsilons = [12, 24, 32, 48]

admixAttack = AdmixAttack(model=cifarResNet, image_width=32, image_height=32)
vmiAttack = VMIAttack(model=cifarResNet)
for eps in epsilons:
    apgdAttack = APGDAttack(predict=cifarResNet, eps=eps / 255, device=device, is_Vt=False)

    if eps < 32:
        steps = torch.arange(0, ((np.ceil(eps / 255 / 0.01))*0.01) + 0.005, 0.005)
    else:
        steps = torch.arange(0, ((np.ceil(eps / 255 / 0.01))*0.01) + 0.01, 0.01)

    # black box
    autoPGDBBoxRobustAccList = [0 for i in range(len(steps))]
    admixBBoxRobustAccList = [0 for i in range(len(steps))]
    vmiBBoxRobustAccList = [0 for i in range(len(steps))]
    # white box
    autoPGDWBoxRobustAccList = [0 for i in range(len(steps))]
    admixWBoxRobustAccList = [0 for i in range(len(steps))]
    vmiWBoxRobustAccList = [0 for i in range(len(steps))]

    # pests
    admixPestList = []
    apgdPestList = []
    vmiPestList = []

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
            admixFiltered = modify_above_threshold(admixPest, threshold, eps / 255)
            vmiFiltered = modify_above_threshold(vmiPest, threshold, eps / 255)
            apgdFiltered = modify_above_threshold(apgdPest, threshold, eps / 255)

            # new Tensors maybe need to use clamp x-eps, x+eps
            apgdAdvF = torch.clamp((inputs + apgdFiltered), 0, 1).detach().to(device)
            admixAdvF = torch.clamp((inputs + admixFiltered), 0, 1).detach().to(device)
            vmiAdvF = torch.clamp((inputs + vmiFiltered), 0, 1).detach().to(device)

            # Black Box Scenario
            correctAPGDBBFilteredAdv, batch_size = computeAcc(cifarVt, apgdAdvF, targets, is_Vt=True)
            correctAdmixBBFilteredAdv, batch_size = computeAcc(cifarVt, admixAdvF, targets, is_Vt=True)
            correctVMIBBFilteredAdv, batch_size = computeAcc(cifarVt, vmiAdvF, targets, is_Vt=True)

            autoPGDBBoxRobustAccList[index] += correctAPGDBBFilteredAdv
            admixBBoxRobustAccList[index] += correctAdmixBBFilteredAdv
            vmiBBoxRobustAccList[index] += correctVMIBBFilteredAdv

            # White Box Scenario
            correctAPGDWBFilteredAdv, batch_size = computeAcc(cifarResNet, apgdAdvF, targets, is_Vt=False)
            correctAdmixWBFilteredAdv, batch_size = computeAcc(cifarResNet, admixAdvF, targets, is_Vt=False)
            correctVMIWBFilteredAdv, batch_size = computeAcc(cifarResNet, vmiAdvF, targets, is_Vt=False)

            autoPGDWBoxRobustAccList[index] += correctAPGDWBFilteredAdv
            admixWBoxRobustAccList[index] += correctAdmixWBFilteredAdv
            vmiWBoxRobustAccList[index] += correctVMIWBFilteredAdv

            if index == 2:
                admixPFlattened = admixFiltered.view(-1).tolist()
                apgdPFlattened = apgdFiltered.view(-1).tolist()
                vmiPFlattened = vmiFiltered.view(-1).tolist()
                # Extend Lists
                vmiPestList.extend(vmiPFlattened)
                apgdPestList.extend(apgdPFlattened)
                admixPestList.extend(admixPFlattened)

    total = len(floader.dataset)
    # Black Box Scenario
    autoPGDBBoxRobustAccListT = [(100. * aPGDCorrect / total) for aPGDCorrect in autoPGDBBoxRobustAccList]
    admixBBoxRobustAccListT = [(100. * admixCorrect / total) for admixCorrect in admixBBoxRobustAccList]
    vmiBBoxRobustAccListT = [(100. * vmiCorrect / total) for vmiCorrect in vmiBBoxRobustAccList]

    print('autoPGDBlackBoxRobustAccList=', autoPGDBBoxRobustAccListT)
    print('admixBlackBoxRobustAccList=', admixBBoxRobustAccListT)
    print('vmiBlackBoxRobustAccList=', vmiBBoxRobustAccListT)

    # White Box Scenario
    autoPGDWBoxRobustAccListT = [(100. * aPGDCorrect / total) for aPGDCorrect in autoPGDWBoxRobustAccList]
    admixWBoxRobustAccListT = [(100. * admixCorrect / total) for admixCorrect in admixWBoxRobustAccList]
    vmiWBoxRobustAccListT = [(100. * vmiCorrect / total) for vmiCorrect in vmiWBoxRobustAccList]

    print('autoPGDWhiteBoxRobustAccList=', autoPGDWBoxRobustAccListT)
    print('admixWhiteBoxRobustAccList=', admixWBoxRobustAccListT)
    print('vmiWhiteBoxRobustAccList=', vmiWBoxRobustAccListT)

    if eps in wantedEpsilons:
        admix_probs, admix_edges = calculate_pmf_and_bin_edges(eps, admixPestList)
        print('admix_probs=', admix_probs.tolist())
        print('admix_edges=', admix_edges.tolist())
        apgdProbs, apgdBins = calculate_pmf_and_bin_edges(eps, apgdPestList)
        print('apgd_probs=', apgdProbs.tolist())
        print('apgd_edges=', apgdBins.tolist())
        vmiProbs, vmiBins = calculate_pmf_and_bin_edges(eps, vmiPestList)
        print('vmi_probs=', vmiProbs.tolist())
        print('vmi_edges=', vmiBins.tolist())
