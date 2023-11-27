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

imagenetteDS = get_dataset('imagenette', 'test')
loaderIN = torch.utils.data.DataLoader(imagenetteDS, batch_size=40, shuffle=False, num_workers=2)
imagenetteResNet = getPreTrainedModel('ResNet18', 'imagenette')

filteredDS = filterDataSet(imagenetteResNet, loaderIN, False)
floader = torch.utils.data.DataLoader(filteredDS, batch_size=40, shuffle=False, num_workers=2)
epsilons = [4, 8, 12, 16, 24, 32, 48, 80, 104]
wantedEpsilons = [12, 24, 32, 48]

alphas = [2, 0.5, 0.25, 0]
print("starting now")
admixM = AdmixAttackModified(model=imagenetteResNet, image_width=160, image_height=160, image_resize=180)
for alpha in alphas:
    admixAccList = []
    for eps in epsilons:
        vmoCorrect = 0
        total = 0
        admixPestList = []
        for batch_idx, (inputs, targets) in enumerate(floader):
            inputs, targets = inputs.to(device), targets.to(device)
            admixAdv = admixM.runAttack(inputs, targets, eps / 255, alphaL2=alpha)
            admixPest = admixAdv - inputs
            correctAdmixAdv, batch_size = computeAcc(imagenetteResNet, admixAdv, targets)
            flattened_list = admixPest.view(-1).tolist()
            admixPestList.extend(flattened_list)
            vmoCorrect += correctAdmixAdv
            total += batch_size
        admixAcc = 100. * vmoCorrect / total
        admixAccList.append(admixAcc)
        if eps in wantedEpsilons:
            print('alpha=', alpha, 'eps=', eps)
            admixProbs, admixEdges = calculate_pmf_and_bin_edges(eps, admixPestList)
            print('admix_probs=', admixProbs.tolist())
            print('admix_edges=', admixEdges.tolist())
    print('alpha=', alpha, 'admixM_Acc=', admixAccList)
