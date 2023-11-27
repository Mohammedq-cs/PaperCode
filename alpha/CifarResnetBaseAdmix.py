import torch
from attacks.modifiedAdmix import AdmixAttackModified
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

cifarDS = get_dataset('cifar10', 'test')
loaderCifar = torch.utils.data.DataLoader(cifarDS, batch_size=80, shuffle=False, num_workers=2)

cifarResNet = getPreTrainedModel('ResNet18', 'cifar10')
filteredDS = filterDataSet(cifarResNet, loaderCifar, False)
floader = torch.utils.data.DataLoader(filteredDS, batch_size=80, shuffle=False, num_workers=2)
epsilons = [12, 24, 32, 48, 64, 80, 104]

admixAccList = []
admixPestList = []
for eps in epsilons:
    print('eps=', eps)
    admixCorrect = 0
    total = 0
    admixAttackM = AdmixAttackModified(model=cifarResNet, l2Alpha=0.001, eps=eps/255, image_width=32, image_height=32)
    for batch_idx, (inputs, targets) in enumerate(floader):
        inputs, targets = inputs.to(device), targets.to(device)
        admixAdv = admixAttackM.runAttack(inputs, targets)
        admixPest = admixAdv - inputs
        correctAdmixAdv, batch_size = computeAcc(cifarResNet, admixAdv, targets)
        flattened_list = admixPest.view(-1).tolist()
        admixPestList.extend(flattened_list)
        admixCorrect += correctAdmixAdv
        total += batch_size
    admixAcc = 100. * admixCorrect / total
    admixAccList.append(admixAcc)
    admixProbs, admixEdges = calculate_pmf_and_bin_edges(eps, admixPestList)
    print('admix probs=', admixProbs.tolist())
    print('admix edges=', admixEdges.tolist())
print('admixM_Acc=', admixAccList)
