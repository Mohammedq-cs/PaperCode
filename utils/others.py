import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def computeAcc(model, inputs, targets, is_Vt=False):
    model.eval()
    outputs = model(inputs)
    if is_Vt:
        outputs = outputs.sup
    _, predicted = outputs.max(1)
    batch_size = targets.size(0)
    correct = predicted.eq(targets).sum().item()
    return correct, batch_size


def computeErrTwoModels(surrogateModel, targetModel, inputs, targets, surrogateIsVT=False, targetIsVT=False):
    surrogateModel.eval()
    targetModel.eval()

    outputsSurrogate = surrogateModel(inputs)
    outputsTarget = targetModel(inputs)

    if surrogateIsVT:
        outputsSurrogate = outputsSurrogate.sup
    if targetIsVT:
        outputsTarget = outputsTarget.sup

    _, predictedSurrogate = outputsSurrogate.max(1)
    _, predictedTarget = outputsTarget.max(1)

    correct_mask = (predictedSurrogate != targets) & (predictedTarget != targets)
    correct = correct_mask.sum().item()
    batch_size = (predictedSurrogate != targets).sum().item()

    return correct, batch_size


def test(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            '''print(batch_idx, len(testloaderMNIST), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))'''

    # Save checkpoint.
    acc = 100. * correct / total
    print(acc)
