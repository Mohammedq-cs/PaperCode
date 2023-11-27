import torch
from attacks.admix import AdmixAttack
from attacks.vmi import VMIAttack
from utils.datasets import get_dataset
from utils.attackUtils import filterDataSetForTwoModels
from utils.loadModels import getPreTrainedModel
import numpy as np


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
loaderCifar = torch.utils.data.DataLoader(cifarDS, batch_size=80, shuffle=False, num_workers=2)
cifarVt = getPreTrainedModel('Visual Transformer', 'cifar10')
cifarResNet = getPreTrainedModel('ResNet18', 'cifar10')

filteredDS = filterDataSetForTwoModels(cifarResNet, cifarVt, loaderCifar, False, True)
floader = torch.utils.data.DataLoader(filteredDS, batch_size=80, shuffle=False, num_workers=2)

epsilons = [12, 24, 32, 48, 80]
admixAttack = AdmixAttack(model=cifarResNet, image_width=32, image_height=32, num_iter=80)
vmiAttack = VMIAttack(model=cifarResNet, num_iters=80)

for eps in epsilons:
    admixCorrect = 0
    total = 0
    aPGDCorrect = 0
    pests = []
    vmiPestList = []
    print("eps=", eps)
    for batch_idx, (inputs, targets) in enumerate(floader):
        inputs, targets = inputs.to(device), targets.to(device)
        admixAdv = admixAttack.runAttack(inputs, targets, eps / 255)
        vmiAdv = vmiAttack.runAttack(inputs, targets, eps / 255)
        pest = admixAdv - inputs
        vmiPest = vmiAdv - inputs
        flattened_list = pest.view(-1).tolist()
        pests.extend(flattened_list)
        vmiPFlattened = vmiPest.view(-1).tolist()
        vmiPestList.extend(vmiPFlattened)

    probs, edges = calculate_pmf_and_bin_edges(eps, pests)
    print('admix_probs=', probs.tolist())
    print('admix_edges=', edges.tolist())
    vmiProbs, vmiBins = calculate_pmf_and_bin_edges(eps, vmiPestList)
    print('vmi_probs=', vmiProbs.tolist())
    print('vmi_edges=', vmiBins.tolist())
