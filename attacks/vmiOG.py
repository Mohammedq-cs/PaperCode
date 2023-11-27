import numpy as np
import torch
import torch.nn as nn

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


def VMI_DI_TI_SI_FGSM(model, images, labels, vmi_cfg):  # we are using this
    r"""
    Overridden.
    """
    beta = vmi_cfg['beta']
    eps = vmi_cfg['epsilon']
    alpha = vmi_cfg['alpha']

    images = images.clone().detach()
    labels = labels.clone().detach()
    batch, channel, H, W = images.shape
    ds = vmi_cfg['dataset']
    if 'cifar' in ds or 'mnist' in ds:
        kernel_size = 3
    elif 'image' in ds:
        kernel_size = 7
    kernel = gkern(kernel_size, 3).astype(np.float32)
    stack_kernel = np.stack([kernel] * channel)
    stack_kernel = np.expand_dims(stack_kernel, 0)
    stack_kernel = torch.tensor(stack_kernel).to(device)
    stack_kernel = stack_kernel.repeat(batch, 1, 1, 1)
    iteration, __, H_Kernel, W_Kernel = stack_kernel.shape
    stack_kernel = stack_kernel.transpose(0, 1)
    stack_kernel = stack_kernel.reshape([batch * channel, 1, H_Kernel, W_Kernel])
    momentum = torch.zeros_like(images).detach()

    v = torch.zeros_like(images).detach()

    loss = nn.CrossEntropyLoss()

    adv_images = images.clone().detach()
    labels = torch.cat(tuple([labels] * 5))
    for _ in range(vmi_cfg['niters']):
        adv_images.requires_grad = True
        si_adv_images = torch.cat(tuple([adv_images, adv_images / 2, adv_images / 4, adv_images / 8, adv_images / 16]), axis=0)

        outputs = model(input_diversity(si_adv_images))
        # set_trace()
        if type(outputs) != list:
            cost = loss(outputs, labels)
        else:
            cost = loss(outputs[0], labels)
        # Update adversarial images

        adv_grad = torch.autograd.grad(cost, adv_images,
                                       retain_graph=False, create_graph=False)[0]
        intimate_grad = adv_grad + v

        current_grad = intimate_grad.reshape([1, batch * channel, H, W])
        current_grad = nn.functional.conv2d(current_grad, stack_kernel, padding='same', groups=channel * batch)
        current_grad = current_grad.reshape([batch, channel, H, W])

        grad = (current_grad) / torch.mean(torch.abs(current_grad), dim=(1, 2, 3), keepdim=True)
        grad = grad + momentum * vmi_cfg['momentum']
        momentum = grad

        # Calculate Gradient Variance
        GV_grad = torch.zeros_like(images).detach()

        for _ in range(vmi_cfg['N']):
            neighbor_images = adv_images.detach() + \
                              torch.randn_like(images).uniform_(eps * beta, eps * beta)
            neighbor_images.requires_grad = True
            input = torch.cat(tuple([neighbor_images, neighbor_images / 2, neighbor_images / 4, neighbor_images / 8, neighbor_images / 16]), axis=0)

            outputs = model(input_diversity(input))

            if type(outputs) != list:
                cost = loss(outputs, labels)
            else:
                cost = loss(outputs[0], labels)

            GV_grad += torch.autograd.grad(cost, neighbor_images,
                                           retain_graph=False, create_graph=False)[0]
        # obtaining the gradient variance

        v = GV_grad / vmi_cfg['N'] - adv_grad

        adv_images = adv_images.detach() + alpha * grad.sign()
        delta = torch.clamp(adv_images - images, min=-eps, max=eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()
    # assert rounddown(adv_images.max().item())<=1 and rounddown(adv_images.min().item())>=0
    # assert rounddown((adv_images - images).min().item())>= -eps and rounddown((adv_images-images).max())<=eps
    return adv_images
