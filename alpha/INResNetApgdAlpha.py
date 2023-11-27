import torch
from attacks.modifiedAutoPGD import APGDAttackModified
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
loaderIN = torch.utils.data.DataLoader(imagenetteDS, batch_size=128, shuffle=False, num_workers=2)
imagenetteResNet = getPreTrainedModel('ResNet18', 'imagenette')

filteredDS = filterDataSet(imagenetteResNet, loaderIN, False)
floader = torch.utils.data.DataLoader(filteredDS, batch_size=128, shuffle=False, num_workers=2)
epsilons = [4, 8, 12, 16, 24, 32, 48, 80, 104]
wantedEpsilons = [12, 24, 32, 48, 80]

alphas = [150, 100, 50, 25, 10, 2, 0, 200]

print("starting now")
for alpha in alphas:
    apgdAccList = []
    for eps in epsilons:
        apgdM = APGDAttackModified(predict=imagenetteResNet, eps=eps / 255, device=device, alphaL2=alpha, is_Vt=False)
        apgdCorrect = 0
        total = 0
        apgdPestList = []
        for batch_idx, (inputs, targets) in enumerate(floader):
            inputs, targets = inputs.to(device), targets.to(device)
            apgdAdv = apgdM.perturb(inputs, targets)
            apgdPest = apgdAdv - inputs
            correctApgdAdv, batch_size = computeAcc(imagenetteResNet, apgdAdv, targets)
            flattened_list = apgdPest.view(-1).tolist()
            apgdPestList.extend(flattened_list)
            apgdCorrect += correctApgdAdv
            total += batch_size
        apgdAcc = 100. * apgdCorrect / total
        apgdAccList.append(apgdAcc)
        if eps in wantedEpsilons:
            print('alpha=', alpha, 'eps=', eps)
            apgdProbs, apgdEdges = calculate_pmf_and_bin_edges(eps, apgdPestList)
            print('apgd_probs=', apgdProbs.tolist())
            print('apgd_edges=', apgdEdges.tolist())
    print('alpha=', alpha, 'apgdM_Acc=', apgdAccList)
