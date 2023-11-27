import torch
import numpy as np
import scipy.stats as st
import torch.nn.functional as F
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AdmixAttackModified:
    def __init__(self, model, image_width, image_height, criterion=torch.nn.CrossEntropyLoss(), num_iter=10, in_channels=3, momentum=0.9, portion=0.2,
                 size=3, prob=0.5, image_resize=45, is_VT=False):
        self.model = model
        self.num_iter = num_iter
        self.momentum = momentum
        self.portion = portion
        self.size = size
        self.image_width = image_width
        self.image_height = image_height
        self.prob = prob
        self.image_resize = image_resize
        self.in_channels = in_channels
        self.is_VT = is_VT
        self.criterion = criterion

    def gkern(self, kernlen=21, nsig=3):
        """Returns a 2D Gaussian kernel array."""
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def getConvFilter(self):
        kernel = self.gkern(7, 3).astype(np.float32)
        stack_kernel = np.stack([kernel] * self.in_channels)
        stack_kernel = np.expand_dims(stack_kernel, 0)
        stack_kernel = torch.tensor(stack_kernel, dtype=torch.float32)
        return stack_kernel

    def input_diversity(self, input_tensor):
        rnd = (torch.randint(self.image_width, self.image_resize, (1,)))
        rescaled = F.interpolate(input_tensor, size=(rnd.item(), rnd.item()), mode='nearest')
        h_rem = self.image_resize - rnd.item()
        w_rem = self.image_resize - rnd.item()
        pad_top = torch.randint(0, h_rem, (1,))
        pad_bottom = h_rem - pad_top.item()
        pad_left = torch.randint(0, w_rem, (1,))
        pad_right = w_rem - pad_left.item()
        padded = F.pad(rescaled, (pad_left.item(), pad_right, pad_top.item(), pad_bottom), value=0.)
        padded = F.pad(padded, (0, 0, 0, 0, 0, 0, 0, 0), value=0.)  # To match the shape
        ret = padded if torch.rand(1) < self.prob else input_tensor
        ret = F.interpolate(ret, size=(self.image_height, self.image_width), mode='nearest')
        return ret

    def admix(self, x, inp_base):
        indices = torch.arange(0, x.shape[0], dtype=torch.int32)
        results = []
        inp_base_tmp = []
        for _ in range(self.size):
            permuted_x = x[torch.randperm(x.shape[0])]
            result_tensor = x + self.portion * permuted_x
            results_inp = inp_base + self.portion * permuted_x
            results.append(result_tensor)
            inp_base_tmp.append(results_inp)
        concatenated_results = torch.cat(results, dim=0)
        concatenated_inp = torch.cat(inp_base_tmp, dim=0)
        return concatenated_results, concatenated_inp

    def runAttack(self, x_inp, y, eps, alphaL2):
        self.model.eval()
        self.model.requires_grad_(False)
        adv_images = x_inp.clone().detach().to(device)
        x_copy = x_inp.clone().detach().to(device)
        alpha = eps / self.num_iter
        prev_grad = torch.zeros(x_inp.size(), dtype=torch.float32).to(device)

        for _ in range(self.num_iter):
            # caculate admix images
            adv_images_admix, x_copy_admix = self.admix(adv_images, x_copy)
            adv_images_batch = torch.cat(
                [adv_images_admix, adv_images_admix / 2., adv_images_admix / 4., adv_images_admix / 8., adv_images_admix / 16.], dim=0)
            adv_images_batch.requires_grad = True
            adv_images_div = self.input_diversity(adv_images_batch)

            # calculate the gradient
            outputs = self.model(adv_images_div).to(device)
            if self.is_VT:
                outputs = outputs.sup
            y_scaled = torch.cat([y] * 5 * self.size, dim=0).to(device)
            ceLoss = self.criterion(outputs, y_scaled)
            x_copy_scaled = torch.cat([x_copy_admix, x_copy_admix / 2., x_copy_admix / 4., x_copy_admix / 8., x_copy_admix / 16.], dim=0).to(device)
            pests = (adv_images_batch - x_copy_scaled).to(device)
            l2Loss = torch.norm(pests, p=2)
            loss = ceLoss + alphaL2 * l2Loss
            loss = torch.mean(loss)
            loss.backward()
            gradients = ((adv_images_batch.grad)).to(device)

            # calculate the mean and sum in Eq 3
            chunks = torch.stack(torch.chunk(gradients, chunks=5, dim=0))
            scaling_factors = (torch.tensor([1, 1 / 2., 1 / 4., 1 / 8., 1 / 16.]).view(5, 1, 1, 1, 1)).to(device)
            noise = torch.mean((chunks * scaling_factors), dim=0)
            noise = torch.sum(torch.stack(torch.chunk(noise, chunks=self.size, dim=0)), dim=0)

            # apply the conv
            stack_kernel = self.getConvFilter().to(device)
            noise = F.conv2d(noise, stack_kernel, stride=[1, 1], padding='same')

            # print(noise.size())
            # update the variables

            noise = noise / torch.mean(torch.abs(noise), dim=[1, 2, 3], keepdim=True).to(device)
            noise = self.momentum * prev_grad + noise
            # update prev grad
            prev_grad = noise

            # update the adv examples
            adv_images = adv_images + alpha * torch.sign(noise)
            adv_images = torch.clamp(adv_images, x_copy - eps, x_copy + eps)
            adv_images = torch.clamp(adv_images, 0, 1).detach()
        return adv_images


class AdmixAttackMixedOg:
    def __init__(self, model, criterion=torch.nn.CrossEntropyLoss(), num_iter=10, in_channels=3, momentum=0.9, portion=0.2,
                 size=3, prob=0.5, is_VT=False):
        self.model = model
        self.num_iter = num_iter
        self.momentum = momentum
        self.portion = portion
        self.size = size
        self.prob = prob
        self.in_channels = in_channels
        self.is_VT = is_VT
        self.criterion = criterion

    def gkern(self, kernlen=21, nsig=3):
        """Returns a 2D Gaussian kernel array."""
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        return kernel

    def getConvFilter(self):
        kernel = self.gkern(7, 3).astype(np.float32)
        stack_kernel = np.stack([kernel] * self.in_channels)
        stack_kernel = np.expand_dims(stack_kernel, 0)
        stack_kernel = torch.tensor(stack_kernel, dtype=torch.float32)
        return stack_kernel

    def input_diversity(self, X, p=0.5):
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

    def admix(self, x):
        indices = torch.arange(0, x.shape[0], dtype=torch.int32)
        results = [x + self.portion * x[torch.randperm(x.shape[0])] for _ in range(self.size)]
        concatenated_results = torch.cat(results, dim=0)
        return concatenated_results

    def runAttack(self, x_inp, y, eps):
        self.model.eval()
        self.model.requires_grad_(False)
        adv_images = x_inp.clone().detach().to(device)
        x_copy = x_inp.clone().detach().to(device)
        alpha = eps / self.num_iter
        prev_grad = torch.zeros(x_inp.size(), dtype=torch.float32).to(device)

        for _ in range(self.num_iter):
            # caculate admix images
            adv_images_admix = self.admix(adv_images)
            adv_images_batch = torch.cat(
                [adv_images_admix, adv_images_admix / 2., adv_images_admix / 4., adv_images_admix / 8., adv_images_admix / 16.], dim=0)
            adv_images_batch.requires_grad = True
            adv_images_div = self.input_diversity(adv_images_batch)

            # calculate the gradient
            outputs = self.model(adv_images_div).to(device)
            if self.is_VT:
                outputs = outputs.sup
            y_scaled = torch.cat([y] * 5 * self.size, dim=0).to(device)
            loss = self.criterion(outputs, y_scaled)
            loss = torch.mean(loss)
            loss.backward()
            gradients = ((adv_images_batch.grad)).to(device)

            # calculate the mean and sum in Eq 3
            chunks = torch.stack(torch.chunk(gradients, chunks=5, dim=0))
            scaling_factors = (torch.tensor([1, 1 / 2., 1 / 4., 1 / 8., 1 / 16.]).view(5, 1, 1, 1, 1)).to(device)
            noise = torch.mean((chunks * scaling_factors), dim=0)
            noise = torch.sum(torch.stack(torch.chunk(noise, chunks=self.size, dim=0)), dim=0)

            # apply the conv
            stack_kernel = self.getConvFilter().to(device)
            noise = F.conv2d(noise, stack_kernel, stride=[1, 1], padding='same')

            # print(noise.size())
            # update the variables

            noise = noise / torch.mean(torch.abs(noise), dim=[1, 2, 3], keepdim=True).to(device)
            noise = self.momentum * prev_grad + noise
            # update prev grad
            prev_grad = noise

            # update the adv examples
            adv_images = adv_images + alpha * torch.sign(noise)
            adv_images = torch.clamp(adv_images, x_copy - eps, x_copy + eps)
            adv_images = torch.clamp(adv_images, 0, 1).detach()
        return adv_images

