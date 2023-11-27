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

epsilons = [4, 8, 12, 16, 24, 32, 48, 64, 80, 104, 128, 160, 192, 240]
wantedEpsilons = [12, 24, 32, 48, 80]

cifarDS = get_dataset('cifar10', 'test')
loaderCifar = torch.utils.data.DataLoader(cifarDS, batch_size=128, shuffle=False, num_workers=2)
cifarCNN = getPreTrainedModel('CNN', 'cifar10')
print("starting experiment 3 CNN ->  ResNet18PORT")
cifarPORT = getPreTrainedModel('ResNet18-PORT', 'cifar10').to(device)
exp3filteredDS = filterDataSetForTwoModels(cifarCNN, cifarPORT, loaderCifar, False, False)
floader = torch.utils.data.DataLoader(exp3filteredDS, batch_size=128, shuffle=False, num_workers=2)

admixAttack = AdmixAttack(model=cifarCNN, image_width=32, image_height=32)
vmiAttack = VMIAttack(model=cifarCNN)
autoPGDImprovedErrList = []
admixImprovedErrList = []
vmiImprovedErrList = []
maxDecErrList = []
maxImprovedErrList = []
for eps in epsilons:
    apgdAttack = APGDAttack(predict=cifarCNN, eps=eps / 255, device=device, is_Vt=False)
    aPGDImprovedCorrect = 0
    admixImprovedCorrect = 0
    vmiImprovedCorrect = 0
    maxDecCorrect = 0
    total = 0
    admixPestList = []
    apgdPestList = []
    vmiPestList = []
    maxDecList = []
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

        # new attack
        admixFiltered = modify_above_threshold(admixPest, 0, eps / 255)
        vmiFiltered = modify_above_threshold(vmiPest, 0, eps / 255)
        apgdFiltered = modify_above_threshold(apgdPest, 0, eps / 255)

        # new Tensors use clamp x-eps, x+eps
        apgdAdvFt = torch.clamp((inputs + apgdFiltered), inputs - (eps / 255), inputs + (eps / 255)).detach()
        admixAdvFt = torch.clamp((inputs + admixFiltered), inputs - (eps / 255), inputs + (eps / 255)).detach()
        vmiAdvFt = torch.clamp((inputs + vmiFiltered), inputs - (eps / 255), inputs + (eps / 255)).detach()

        apgdAdvF = torch.clamp(apgdAdvFt, 0, 1).detach().to(device)
        admixAdvF = torch.clamp(admixAdvFt, 0, 1).detach().to(device)
        vmiAdvF = torch.clamp(vmiAdvFt, 0, 1).detach().to(device)

        # calculate Acc
        correctAPGDAdv, batch_size = computeAcc(cifarPORT, apgdAdvF, targets, is_Vt=False)
        correctAdmixAdv, batch_size = computeAcc(cifarPORT, admixAdvF, targets, is_Vt=False)
        correctVMIAdv, batch_size = computeAcc(cifarPORT, vmiAdvF, targets, is_Vt=False)
        correctMaxDecAdv, batch_size = computeAcc(cifarPORT, maxDecAdv, targets, is_Vt=False)
        aPGDImprovedCorrect += correctAPGDAdv
        admixImprovedCorrect += correctAdmixAdv
        vmiImprovedCorrect += correctVMIAdv
        maxDecCorrect += correctMaxDecAdv
        total += batch_size

        # Flatten Lists
        admixPFlattened = admixFiltered.view(-1).tolist()
        apgdPFlattened = vmiFiltered.view(-1).tolist()
        vmiPFlattened = apgdFiltered.view(-1).tolist()
        maxDecFlattened = maxDecPest.view(-1).tolist()
        # Extend Lists
        vmiPestList.extend(vmiPFlattened)
        apgdPestList.extend(apgdPFlattened)
        admixPestList.extend(admixPFlattened)
        maxDecList.extend(maxDecFlattened)

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
        maxDecProbs, maxDecBins = calculate_pmf_and_bin_edges(eps, maxDecList)
        print('maxDec_probs=', maxDecProbs.tolist())
        print('maxDec_edges=', maxDecBins.tolist())

    aPGDCAcc = 100. * aPGDImprovedCorrect / total
    admixAcc = 100. * admixImprovedCorrect / total
    vmiAcc = 100. * vmiImprovedCorrect / total
    maxDecAcc = 100. * maxDecCorrect / total
    maxErr = max([(100 - aPGDCAcc) / 100, (100 - admixAcc) / 100, (100 - vmiAcc) / 100, (100 - maxDecAcc) / 100])
    print('aPGDCAcc=', aPGDCAcc, ' admixAcc=', admixAcc, ' vmiAcc=', vmiAcc, ' maxDecAcc=', maxDecAcc, ' maxErr=', maxErr)
    maxImprovedErrList.append(maxErr)
    autoPGDImprovedErrList.append((100 - aPGDCAcc) / 100)
    admixImprovedErrList.append((100 - admixAcc) / 100)
    vmiImprovedErrList.append((100 - vmiAcc) / 100)
    maxDecErrList.append((100 - maxDecAcc) / 100)

print('autoPGDImprovedErrList=', autoPGDImprovedErrList)
print('admixErrImprovedList=', admixImprovedErrList)
print('vmiErrImprovedList=', vmiImprovedErrList)
print('maxErrImprovedList=', maxImprovedErrList)
print('maxDecErrList=', maxDecErrList)