import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class VMIAttackModified:
    def __init__(self, model, criterion=torch.nn.CrossEntropyLoss(), num_iters=10, num_variance_iters=20, momentum=0.9, num_classes=10, is_VT=False):
        self.model = model
        self.num_iters = num_iters
        self.num_variance_iters = num_variance_iters
        self.momentum = momentum
        self.num_classes = num_classes
        self.is_VT = is_VT
        self.criterion = criterion

    def batcGrad(self, x_adv, y, alpha, grad_size):
        x_copy = x_adv.clone().detach().to(device)
        global_grad = torch.zeros_like(grad_size).to(device)
        # sampling N examples in the neighborhood of x
        for _ in range(self.num_variance_iters):
            x_neighbor = x_copy + torch.FloatTensor(x_copy.size()).uniform_(-alpha, alpha).to(device)
            x_neighbor.requires_grad = True
            outputs = self.model(x_neighbor).to(device)
            if self.is_VT:
                outputs = outputs.sup
            loss = self.criterion(outputs, y)
            loss = torch.mean(loss)
            loss.backward()
            global_grad += x_neighbor.grad

            x_neighbor.grad.zero_()
            x_neighbor = x_neighbor.detach().to(device)
        return global_grad

    def runAttack(self, x_inp, y, eps, alphaL2):
        self.model.eval()
        self.model.requires_grad_(False)
        alpha = eps / self.num_iters
        variance = torch.zeros(x_inp.size()).to(device)
        adv_images = x_inp.clone().detach().to(device)
        x_copy = x_inp.clone().detach().to(device)
        prev_grad = torch.zeros(x_inp.size(), dtype=torch.float32).to(device)
        for _ in range(self.num_iters):
            # calculate the gradient
            adv_images.requires_grad = True
            outputs = self.model(adv_images).to(device)
            if self.is_VT:
                outputs = outputs.sup
            pests = (adv_images - x_copy).to(device)
            l2Loss = torch.norm(torch.norm(pests, p=2))
            ceLoss = self.criterion(outputs, y.to(device))
            loss = ceLoss + alphaL2 * l2Loss
            loss = torch.mean(loss)
            loss.backward()
            new_grad = (adv_images.grad).to(device)

            # update the gradient by variance tuning based momentum
            current_grad = (new_grad + variance).to(device)
            # this is the updated grad
            noise = (self.momentum * prev_grad + (current_grad / torch.mean(torch.abs(current_grad), dim=[1, 2, 3], keepdim=True))).to(device)

            # update the variance
            global_grad = self.batcGrad(adv_images, y, alpha, new_grad)
            variance = (global_grad / 1. * self.num_variance_iters) - new_grad

            # update prev gead
            prev_grad = noise
            # update x_adv by fgsm
            adv_images.grad.zero_()
            adv_images = adv_images.detach() + alpha * torch.sign(noise)
            adv_images = torch.clamp(adv_images, x_copy - eps, x_copy + eps)
            # Update the adversarial images for the next iteration
            adv_images = torch.clamp(adv_images.detach(), 0, 1).detach()

        return adv_images
