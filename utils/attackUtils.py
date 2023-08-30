import torch
from torch.utils.data import Subset, DataLoader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch
from torch.utils.data import Subset


def filterDataSetForTwoModels(surrogateModel, targetModel, dataloader, surrogateIsVT=False, targetIsVT=False):
    surrogateModel.eval()
    targetModel.eval()
    correct = 0
    total = 0

    correctly_classified_indices = []

    # Step 1: Iterate through the DataLoader and make predictions
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputsSurrogate = surrogateModel(inputs)
            outputsTarget = targetModel(inputs)

            if surrogateIsVT:
                outputsSurrogate = outputsSurrogate.sup
            if targetIsVT:
                outputsTarget = outputsTarget.sup

            _, predictedSurrogate = outputsSurrogate.max(1)
            _, predictedTargetM = outputsTarget.max(1)

            correct_mask = (predictedSurrogate == targets) & (predictedTargetM == targets)
            total += targets.size(0)
            correct += correct_mask.sum().item()

            correctly_classified_indices.extend(
                (batch_idx * dataloader.batch_size) + i
                for i, is_correct in enumerate(correct_mask)
                if is_correct
            )

    correctly_classified_dataset = Subset(dataloader.dataset, correctly_classified_indices)
    acc = 100.0 * correct / total
    print(acc)

    return correctly_classified_dataset


def filterDataSet(model, dataloader, isVT=False):
    model.eval()
    correct = 0
    total = 0

    correctly_classified_indices = []

    # Step 1: Iterate through the DataLoader and make predictions

    with torch.no_grad():

        for batch_idx, (inputs, targets) in enumerate(dataloader):

            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            if isVT:
                outputs = outputs.sup

            _, predicted = outputs.max(1)

            correct_mask = predicted == targets
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            correctly_classified_indices.extend((batch_idx * dataloader.batch_size) + i for i, is_correct in enumerate(correct_mask) if is_correct)

    correctly_classified_dataset = Subset(dataloader.dataset, correctly_classified_indices)
    acc = 100. * correct / total
    print(acc)

    return correctly_classified_dataset
