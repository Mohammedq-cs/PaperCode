import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable as V
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def input_diversity(X, p=0.5):
    """
    AW: i changed image_width and image_resize to be automatic
    """
    # AW logic fix: learn image_width from X.shape
    _, _, h, w = X.shape
    assert (h == w)
    image_width = h
    image_resize = int(330 / 299 * h)
    # AW optimize: change random to start instead of end
    if torch.rand(()) >= p:
        return X
    rnd = torch.randint(image_width, image_resize, ())
    rescaled = nn.functional.interpolate(X, [rnd, rnd])
    h_rem = image_resize - rnd
    w_rem = image_resize - rnd
    pad_top = torch.randint(0, h_rem, ())
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(0, w_rem, ())
    pad_right = w_rem - pad_left
    padded = nn.ConstantPad2d((pad_left, pad_right, pad_top, pad_bottom), 0.)(rescaled)
    padded = nn.functional.interpolate(padded, [image_width, image_width])
    # return padded if torch.rand(()) < p else X

    return padded


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    import scipy.stats as st

    x = np.linspace(-nsig, nsig, kernlen)
    kern1d = st.norm.pdf(x)
    kernel_raw = np.outer(kern1d, kern1d)
    kernel = kernel_raw / kernel_raw.sum()
    return kernel


def admix(x, size=3):
    portion = 0.2
    # size=3 #mixup
    return torch.cat(tuple([(x + portion * x[torch.randperm(x.size(0))]) for _ in range(size)]), axis=0) / (1 + portion * size)


def zero_gradients(x):
    if x.grad is not None:
        x.grad.zero_()
    return x


def rounddown(number):
    return int(number * 10000) / 10000


def admix_TI_FGSM(model, img, label, cfg, using_aux_logit=False):
    eps = cfg['epsilon']
    num_iter = cfg['niters']
    alpha = eps / num_iter
    momentum = 1  # set in the original paper
    grad = 0
    X_pert = img.clone()
    noise = torch.zeros_like(img, requires_grad=True)
    size = 3
    old_grad = torch.zeros_like(img)
    label = torch.cat(tuple([label] * 3))

    batch, channel, H, W = X_pert.shape
    if 'image' in cfg['dataset']:  # AW
        kernel = gkern(9, 3).astype(np.float32)
    elif cfg['dataset'] in ['mnist', 'cifar']:  # AW
        kernel = gkern(3, 3).astype(np.float32)
    else:
        raise Exception(f'got dataset {cfg["dataset"]}')
    # stack_kernel = np.stack([kernel, kernel, kernel])
    stack_kernel = np.stack([kernel] * channel)  # AW: support different channel numbers
    stack_kernel = np.expand_dims(stack_kernel, 0)
    stack_kernel = torch.tensor(stack_kernel).cuda()
    stack_kernel = stack_kernel.repeat(batch, 1, 1, 1)
    iteration, __, H_Kernel, W_Kernel = stack_kernel.shape
    stack_kernel = stack_kernel.transpose(0, 1)
    stack_kernel = stack_kernel.reshape([batch * channel, 1, H_Kernel, W_Kernel])

    for i in range(num_iter):
        zero_gradients(noise)

        x = admix(X_pert + noise, size)

        x_nes_2 = 1 / 2 * x
        x_nes_4 = 1 / 4 * x
        x_nes_8 = 1 / 8 * x
        x_nes_16 = 1 / 16 * x
        for i, each_x in enumerate([x, x_nes_2, x_nes_4, x_nes_8, x_nes_16]):
            output = model(input_diversity(each_x))
            # import code ; code.interact(local=dict(globals(),**locals()))
            if i == 0:
                loss = F.cross_entropy(output, label)  # logit
            else:
                loss = loss + F.cross_entropy(output, label)  # logit
            if using_aux_logit:
                raise ValueError("AW: I didn't really llok into that yet")
                loss = loss + F.cross_entropy(output[1], label)  # aux_logit
        loss.backward()
        grad = noise.grad.data

        # translation invariant
        grad = grad.reshape([1, batch * channel, H, W])
        grad = nn.functional.conv2d(grad, stack_kernel, padding='same', groups=channel * batch)
        grad = grad.reshape([batch, channel, H, W])

        # momentum
        grad = grad / torch.abs(grad).mean([1, 2, 3], keepdim=True)
        grad = momentum * old_grad + grad
        old_grad = grad

        noise = noise + alpha * torch.sign(grad)
        # Avoid out of bound
        noise = torch.clamp(noise, -eps, eps)
        x = img + noise
        x = torch.clamp(x, 0.0, 1.0)
        noise = x - img

        noise = V(noise, requires_grad=True)

    adv = img + noise.detach()
    # TODO:assert rounddown(adv.max().item())<=1 and rounddown(adv.min().item())>=0A6000
    # TODO:assert(np.all(np.logical_and(adv-img>-eps, adv-img<eps)))
    assert rounddown(adv.max().item()) <= 1 and rounddown(adv.min().item()) >= 0
    assert rounddown((adv - img).min().item()) >= -eps and rounddown((adv - img).max()) <= eps

    return adv
