import torch
from attacks.modifiedVmi import VMIAttackModified
from utils.datasets import get_dataset
from utils.attackUtils import filterDataSet
from utils.loadModels import getPreTrainedModel
from utils.others import computeAcc
import numpy as np
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
def calculate_pmf_and_bin_edges(eps, values, bin_width=0.01):
    min_value = np.floor(-eps / 255 / bin_width) * bin_width
    max_value = np.ceil(eps / 255 / bin_width) * bin_width
    bin_edges = np.arange(min_value, max_value + bin_width, bin_width)

    counts, _ = np.histogram(values, bins=bin_edges, density=True)
    probs = counts / float(counts.sum())

    return probs, bin_edges[:-1]

imagenetteDS = get_dataset('imagenette', 'test')
loaderIN = torch.utils.data.DataLoader(imagenetteDS, batch_size=90, shuffle=False, num_workers=2)
imagenetteResNet = getPreTrainedModel('ResNet18', 'imagenette')

filteredDS = filterDataSet(imagenetteResNet, loaderIN, False)
floader = torch.utils.data.DataLoader(filteredDS, batch_size=90, shuffle=False, num_workers=2)
epsilons = [4, 8, 12, 16, 24, 32, 48, 80, 104]
wantedEpsilons = [12, 24, 32, 48]

alphas = [10000, 0, 500, 50, 1, 100000]
print("starting now")
vmiM = VMIAttackModified(model=imagenetteResNet)
for alpha in alphas:
    vmiAccList = []
    for eps in epsilons:
        vmoCorrect = 0
        total = 0
        vmiPestList = []
        for batch_idx, (inputs, targets) in enumerate(floader):
            inputs, targets = inputs.to(device), targets.to(device)
            vmiAdv = vmiM.runAttack(inputs, targets, eps/255, alphaL2=alpha)
            vmiPest = vmiAdv - inputs
            correctVmiAdv, batch_size = computeAcc(imagenetteResNet, vmiAdv, targets)
            flattened_list = vmiPest.view(-1).tolist()
            vmiPestList.extend(flattened_list)
            vmoCorrect += correctVmiAdv
            total += batch_size
        vmiAcc = 100. * vmoCorrect / total
        vmiAccList.append(vmiAcc)
        if eps in wantedEpsilons:
            print('alpha=', alpha, 'eps=', eps)
            vmiProbs, vmiEdges = calculate_pmf_and_bin_edges(eps, vmiPestList)
            print('vmi_probs=', vmiProbs.tolist())
            print('vmi_edges=', vmiEdges.tolist())
    print('alpha=', alpha, 'vmiM_Acc=', vmiAccList)
