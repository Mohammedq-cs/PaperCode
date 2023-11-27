import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class AdmixAttackLiteModified:
    def __init__(self, model, image_width, image_height, num_iter=10, in_channels=3, momentum=0.9, portion=0.2, size=3, prob=0.5, image_resize=45,
                 is_VT=False):
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
        self.criterion = torch.nn.CrossEntropyLoss()

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

            # calculate the gradient
            outputs = self.model(adv_images_batch).to(device)
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
            gradients = adv_images_batch.grad.to(device)

            # calculate the mean and sum in Eq 3
            chunks = torch.stack(torch.chunk(gradients, chunks=5, dim=0))
            scaling_factors = (torch.tensor([1, 1 / 2., 1 / 4., 1 / 8., 1 / 16.]).view(5, 1, 1, 1, 1)).to(device)
            noise = torch.mean((chunks * scaling_factors), dim=0)
            noise = torch.sum(torch.stack(torch.chunk(noise, chunks=self.size, dim=0)), dim=0)

            noise = noise / torch.mean(torch.abs(noise), dim=[1, 2, 3], keepdim=True).to(device)
            noise = self.momentum * prev_grad + noise
            # update prev grad
            prev_grad = noise
            adv_images_batch.grad.zero_()
            # update the adv examples
            adv_images = adv_images + alpha * torch.sign(noise)
            adv_images = torch.clamp(adv_images, x_copy - eps, x_copy + eps)
            adv_images = torch.clamp(adv_images, 0, 1).detach()
        return adv_images
