from .torchtoolbox_aw_transforms import ImageNetPolicy
from torchvision.transforms import  ToPILImage

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms as T
import torch.nn.functional as F
from torch.autograd import Variable as V
import math
# from torch.autograd.gradcheck import zero_gradients
from torch.utils import data
import os
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
# from ipdb import  set_trace

# from PIL import Image, ImageFilter, ImageGrab
from torchvision import transforms
# from utils import regularizer,rand_bbox
# from colourspace import  group_pca_color_augmention

# AW: instead of importing torchtoolbox:
# TODO


def zero_gradients(x):
    if x.grad is not None:
        x.grad.zero_()


    return x


list_nets = [
    'tf_inception_v3',
    'tf_inception_v4',
    'tf_resnet_v2_50',

    'tf_resnet_v2_152',
    'tf_inc_res_v2',
    'tf_resnet_v2_101',
    'tf_adv_inception_v3',
    'tf_ens3_adv_inc_v3',
    'tf_ens4_adv_inc_v3',
    'tf_ens_adv_inc_res_v2',

    ]
#torch.backends.cudnn.enabled = False
parser = argparse.ArgumentParser()
parser.add_argument('--method', type=str, default='benign', help='the attack method used')
parser.add_argument('--visualize', type=bool, default=False, help="whether to visualize the discriminitive  region")

parser.add_argument('--gpu', type=str, default='0', help='The ID of GPU to use.')
parser.add_argument('--input_csv', type=str, default='dataset/dev_dataset.csv', help='Input csv with images.')
parser.add_argument('--input_dir', type=str, default='dataset/images/', help='Input images.')
parser.add_argument('--output_dir', type=str, default='adv_img_torch/', help='Output directory with adv images.')
parser.add_argument('--model_dir', type=str, default='torch_nets_weight/', help='Model weight directory.')

parser.add_argument("--max_epsilon", type=float, default=16.0, help="Maximum size of adversarial perturbation.")
parser.add_argument("--num_iter", type=int, default=10, help="Number of iterations.")
parser.add_argument("--batch_size", type=int, default=10, help="How many images process at one time.")

parser.add_argument("--momentum", type=float, default=1, help="Momentum")

# opt = parser.parse_args()

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
# device=
# device = torch.device("cuda:"+opt.gpu)

def seed_torch(seed):
    """Set a random seed to ensure that the results are reproducible"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def mkdir(path):
    """Check if the folder exists, if it does not exist, create it"""
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)


class Normalize(nn.Module):

    def __init__(self, mean=0, std=1, mode='tensorflow'):
        """
        mode:
            'tensorflow':convert data from [0,1] to [-1,1]
            'torch':(input - mean) / std
        """
        super(Normalize, self).__init__()
        self.mean = mean
        self.std = std
        self.mode = mode

    def forward(self, input):
        size = input.size()
        x = input.clone()

        if self.mode == 'tensorflow':
            x = x * 2.0 - 1.0  # convert data from [0,1] to [-1,1]
        elif self.mode == 'torch':
            for i in range(size[1]):
                x[:, i] = (x[:, i] - self.mean[i]) / self.std[i]
        return x


class ImageNet(data.Dataset):
    """load data from img and csv"""
    def __init__(self, dir, csv_path, transforms=None):
        self.dir = dir
        self.csv = pd.read_csv(csv_path)
        self.transforms = transforms

    def __getitem__(self, index):
        img_obj = self.csv.loc[index]
        ImageID = img_obj['ImageId'] + '.png'
        Truelabel = img_obj['TrueLabel']
        img_path = os.path.join(self.dir, ImageID)
        pil_img = Image.open(img_path).convert('RGB')
        if self.transforms:
            data = self.transforms(pil_img)
        else:
            data = pil_img
        return data, ImageID, Truelabel

    def __len__(self):
        return len(self.csv)


def get_model(net_name, model_dir):
    """Load converted model"""
    model_path = os.path.join(model_dir, net_name + '.npy')

    if net_name == 'tf_inception_v3':
        net = tf_inception_v3
    elif net_name == 'tf_inception_v4':
        net = tf_inception_v4
    elif net_name == 'tf_resnet_v2_50':
        net = tf_resnet_v2_50
    elif net_name == 'tf_resnet_v2_101':
        net = tf_resnet_v2_101
    elif net_name == 'tf_resnet_v2_152':
        net = tf_resnet_v2_152
    elif net_name == 'tf_inc_res_v2':
        net = tf_inc_res_v2
    elif net_name == 'tf_adv_inception_v3':
        net = tf_adv_inception_v3
    elif net_name == 'tf_ens3_adv_inc_v3':
        net = tf_ens3_adv_inc_v3
    elif net_name == 'tf_ens4_adv_inc_v3':
        net = tf_ens4_adv_inc_v3
    elif net_name == 'tf_ens_adv_inc_res_v2':
        net = tf_ens_adv_inc_res_v2
    else:
        print('Wrong model name!')

    model = nn.Sequential(
        # Images for inception classifier are normalized to be in [-1, 1] interval.
        Normalize('tensorflow'),
        net.KitModel(model_path).eval().cuda(),)
    return model
def rounddown(number):
    return int(number * 10000) / 10000

def get_models(list_nets, model_dir):
    """load models with dict"""
    nets = {}
    for net in list_nets:
        nets[net] = get_model(net, model_dir)
    return nets


def save_img(images, filenames, output_dir):
    """save high quality jpeg"""
    mkdir(output_dir)

    for i, filename in enumerate(filenames):

        # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
        ndarr = images[i].mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
        img = Image.fromarray(ndarr)
        img.save(os.path.join(output_dir, filename), quality=100)

def IFGSM(model, img, label,using_aux_logit):
    eps = opt.max_epsilon / 255.0
    num_iter = opt.num_iter
    alpha = eps / num_iter
    momentum = opt.momentum

    noise = torch.zeros_like(img, requires_grad=True)

    old_grad = 0.0
    for i in range(num_iter):
        zero_gradients(noise)
        x = img + noise
        output = model(x)

        loss = F.cross_entropy(output[0], label)  # logit
        if using_aux_logit:
            loss += F.cross_entropy(output[1], label)  # aux_logit
        loss.backward()
        grad = noise.grad.data
        # MI-FGSM
        # grad = grad / torch.abs(grad).mean([1,2,3], keepdim=True)
        # grad = momentum * old_grad + grad
        # old_grad = grad

        noise = noise + alpha * torch.sign(grad)
        # Avoid out of bound
        noise = torch.clamp(noise, -eps, eps)
        x = img + noise
        x = torch.clamp(x, 0.0, 1.0)
        noise = x - img
        noise = V(noise, requires_grad=True)

    adv = img + noise.detach()
    assert rounddown(adv.max().item())<=1 and rounddown(adv.min().item())>=0
    assert rounddown((adv - img).min().item())>= -eps and rounddown((adv - img).max())<=eps

    if opt.visualize:
        for index in range(opt.batch_size):
            import time
            with torch.no_grad():
                time_start = time.time()
                output = model(torch.unsqueeze(img[index],0))
                time_2 = time.time()
                print(f"classify the origin image:{time_2-time_start}")
                cam = generate_cam(model, module_name, features_blobs, torch.unsqueeze(img[index],0), target_class=torch.argmax(output[0]),device=device)
                time_3 = time.time()

                print(f"generate the cmp for original image{time_3-time_2}")
                save_class_activation_images(img[index], cam, f'benign_{use_model}_{index}_{torch.argmax(output[0])}', dictionary='IFGSM')
                time_4 = time.time()
                adversarial = model(torch.unsqueeze(adv[index],0))
                time_5= time.time()
                print(f"classify the adv mage:{time_5-time_4}")
                cam = generate_cam(model, module_name, features_blobs, torch.unsqueeze(adv[index],0), target_class=torch.argmax(adversarial[0]),device=device)
                time_6= time.time()
                print(f"generate the cmp for adv image{time_6-time_5}")
                save_class_activation_images(adv[index], cam, f'{use_model}_{index}_{torch.argmax(adversarial[0])}', dictionary='IFGSM')


    return adv
def MIFGSM(model, img, label,using_aux_logit=True):
    eps = opt.max_epsilon / 255.0
    num_iter = opt.num_iter
    alpha = eps / num_iter
    momentum = opt.momentum

    noise = torch.zeros_like(img, requires_grad=True)

    old_grad = 0.0

    for i in range(num_iter):
        zero_gradients(noise)
        x = img + noise
        output = model(x)

        loss = F.cross_entropy(output[0], label)  # logit
        if using_aux_logit:
            loss += F.cross_entropy(output[1], label)  # aux_logit

        loss.backward()
        grad = noise.grad.data
        # MI-FGSM
        grad = grad / torch.abs(grad).sum([1,2,3], keepdim=True)
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
    assert rounddown(adv.max().item())<=1 and rounddown(adv.min().item())>=0
    assert rounddown((adv - img).min().item())>= -eps and rounddown((adv - img).max())<=eps
    #visualize
    global imageindex


    return adv







def input_diversity(X, p=0.5) :
    """
    AW: i changed image_width and image_resize to be automatic
    """
    #AW logic fix: learn image_width from X.shape
    _,_,h,w = X.shape
    assert(h==w)
    image_width = h
    image_resize = int(330/299*h)
    #AW optimize: change random to start instead of end
    if torch.rand(()) >= p:
        return X
    rnd = torch.randint(image_width, image_resize, ())
    rescaled = nn.functional.interpolate(X, [rnd, rnd])
    h_rem = image_resize - rnd
    w_rem = image_resize - rnd
    pad_top = torch.randint(0, h_rem, ())
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(0, w_rem,())
    pad_right = w_rem - pad_left
    padded = nn.ConstantPad2d((pad_left, pad_right, pad_top, pad_bottom), 0.)(rescaled)
    padded = nn.functional.interpolate(padded, [image_width, image_width])
    #return padded if torch.rand(()) < p else X

    return padded


def Edge_Enhance(x):
    # AW: since in_channels is somtimes 1, and in_channels/groups should be int, so groups must  be 1.
    _,in_channels,_,_ = x.shape
    if in_channels==1:
        group_num = 1
    elif in_chnanels==3:
        group_num = 3
    kernel=torch.unsqueeze(torch.tensor([[-0.5,-0.5,-0.5],[-0.5,5,-0.5],[-0.5,-0.5,-0.5]]),0)
    kernel=torch.unsqueeze(kernel,dim=0).cuda()
    kernel=torch.repeat_interleave(kernel,group_num,dim=0)

    return F.conv2d(x,kernel,padding='same',groups=group_num)
def CUTOUT_MIFGSM(model, img, label,using_aux_logit):
    eps = opt.max_epsilon / 255.0

    num_iter = opt.num_iter
    alpha = eps / num_iter
    momentum = 1  # set in the original paper
    grad = 0
    X_pert = img.clone()
    noise = torch.zeros_like(img, requires_grad=True)
    old_grad = torch.zeros_like(img)
    # label = torch.cat(tuple([label] * 3))

    for i in range(num_iter):
        zero_gradients(noise)
        # set_trace()
        # x_cs=channel_shuffle((X_pert+ noise))
        # save_img(x_cs, [str(i)+".jpeg" for i in range(x_cs.shape[0])], opt.output_dir)


        x_origin=X_pert + noise
        x_RE1 = mycutout(X_pert + noise, p=0.5, ratio=(1, 1), value=(0, 1))

        for i, each_x in enumerate([x_origin, x_RE1]):
            output = model(each_x)
            # set_trace()
            if i == 0:
                loss = F.cross_entropy(output[0], label)  # logit
            else:
                loss = loss + F.cross_entropy(output[0], label)  # logit
            if using_aux_logit:
                loss = loss + F.cross_entropy(output[1], label)  # aux_logit
        loss.backward()
        grad = noise.grad.data

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
def CUTMIX_MIFGSM(model, img, label,using_aux_logit):
    eps = opt.max_epsilon / 255.0

    num_iter = opt.num_iter
    alpha = eps / num_iter
    momentum = 1  # set in the original paper
    grad = 0
    X_pert = img.clone()
    noise = torch.zeros_like(img, requires_grad=True)
    old_grad = torch.zeros_like(img)
    # label = torch.cat(tuple([label] * 3))

    for i in range(num_iter):
        zero_gradients(noise)
        # set_trace()
        # x_cs=channel_shuffle((X_pert+ noise))
        # save_img(x_cs, [str(i)+".jpeg" for i in range(x_cs.shape[0])], opt.output_dir)



        rand_index = torch.randperm(img.shape[0]).cuda()
        bbx1, bby1, bbx2, bby2 = rand_bbox(img.shape)
        x =X_pert + noise
        x_origin=X_pert + noise
        x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]


        for i, each_x in enumerate([x_origin, x]):
            output = model(each_x)
            # set_trace()
            if i == 0:
                loss = F.cross_entropy(output[0], label)  # logit
            else:
                loss = loss + F.cross_entropy(output[0], label)  # logit
            if using_aux_logit:
                loss = loss + F.cross_entropy(output[1], label)  # aux_logit
        loss.backward()
        grad = noise.grad.data

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

    assert rounddown(adv.max().item()) <= 1 and rounddown(adv.min().item()) >= 0
    assert rounddown((adv - img).min().item()) >= -eps and rounddown((adv - img).max()) <= eps

    return adv

def SI_NI_TI_DI_FGSM(model, img, label,using_aux_logit):
    eps = opt.max_epsilon / 255.0
    num_iter = opt.num_iter
    alpha = eps / num_iter
    momentum = 1# set in the original paper
    grad=0
    X_pert  = img.clone()
    noise = torch.zeros_like(img, requires_grad=True)

    for i in range(num_iter):
        zero_gradients(noise)
        x=X_pert+ noise+ momentum * alpha * grad
        x_nes_2=1/2*x
        x_nes_4 = 1 / 4 * x
        x_nes_8 = 1 / 8 * x
        x_nes_16 = 1 / 16 * x
        temp_grad=0
        for i,each_x in enumerate([x,x_nes_2 ,x_nes_4,x_nes_8,x_nes_16]):
            zero_gradients(noise)
            output = model(input_diversity(each_x))
            loss = F.cross_entropy(output[0], label)  # logit
            if using_aux_logit:
                loss =loss + F.cross_entropy(output[1], label)  # aux_logit
            loss.backward()
            temp_grad+=noise.grad.data
            # if i ==0:
            #     loss = F.cross_entropy(output[0], label)  # logit
            # else:
            #     loss=loss+ F.cross_entropy(output[0], label)  # logit
            # if using_aux_logit:
            #     loss =loss + F.cross_entropy(output[1], label)  # aux_logit
        # loss.backward()
        grad = temp_grad
        # MI-FGSM
        grad = grad / torch.abs(grad).sum([1,2,3], keepdim=True)
        grad = momentum * old_grad + grad
        old_grad = grad
        # grad = grad / torch.abs(grad).mean([1, 2, 3], keepdim=True)

        noise = noise + alpha * torch.sign(grad)
        # Avoid out of bound
        noise = torch.clamp(noise, -eps, eps)
        x = img + noise
        x = torch.clamp(x, 0.0, 1.0)
        noise = x - img
        noise = V(noise, requires_grad=True)

    adv = img + noise.detach()
    assert rounddown(adv.max().item())<=1 and rounddown(adv.min().item())>=0
    assert rounddown((adv - img).min().item())>= -eps and rounddown((adv - img).max())<=eps
    return adv
def neural_TDSMFGSM(model, img,img_n, label, using_aux_logit):
    eps = opt.max_epsilon / 255.0
    num_iter = opt.num_iter
    alpha = eps / num_iter
    momentum = 1  # set in the original paper
    grad = 0
    X_pert = img.clone()
    x_neural=img_n.clone()
    noise = torch.zeros_like(img, requires_grad=True)
    size = 3
    old_grad = torch.zeros_like(img)
    # label = torch.cat(tuple([label] * 3))

    batch, channel, H, W = X_pert.shape
    kernel = gkern(7, 3).astype(np.float32)
    stack_kernel = np.stack([kernel, kernel, kernel])
    stack_kernel = np.expand_dims(stack_kernel, 0)
    stack_kernel = torch.tensor(stack_kernel).cuda()
    stack_kernel = stack_kernel.repeat(batch, 1, 1, 1)
    iteration, __, H_Kernel, W_Kernel = stack_kernel.shape
    stack_kernel = stack_kernel.transpose(0, 1)
    stack_kernel = stack_kernel.reshape([batch * channel, 1, H_Kernel, W_Kernel])

    for i in range(num_iter):
        zero_gradients(noise)

        # x = admix(X_pert + noise, size)
        x=X_pert + noise
        x_n=x_neural+noise

        x_nes_2 = 1 / 2 * x
        x_nes_4 = 1 / 4 * x
        x_nes_8 = 1 / 8 * x
        x_nes_16 = 1 / 16 * x
        x_n_2=1 / 2 * img_n
        x_n_4 = 1 / 2 *x_n
        x_n_8 = 1 / 2 * x_n
        x_n_16 = 1 / 2 * x_n
        for i, each_x in enumerate([x,img_n, x_nes_2, x_nes_4, x_nes_8, x_nes_16,x_n_2,x_n_4,x_n_8,x_n_16]):
            output = model(input_diversity(each_x))

            if i == 0:
                loss = F.cross_entropy(output[0], label)  # logit
            else:
                loss = loss + F.cross_entropy(output[0], label)  # logit
            if using_aux_logit:
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
def neuraltransfer_MIFGSM(model, img,img_n, label, using_aux_logit):
    eps = opt.max_epsilon / 255.0
    num_iter = opt.num_iter
    alpha = eps / num_iter
    momentum = 1  # set in the original paper
    grad = 0
    X_pert = img.clone()
    x_neural=img_n.clone()
    noise = torch.zeros_like(img, requires_grad=True)
    old_grad = torch.zeros_like(img)


    for i in range(num_iter):
        zero_gradients(noise)

        # x = admix(X_pert + noise, size)
        x=X_pert + noise
        x_n=x_neural+noise


        for i, each_x in enumerate([x,x_n]):
            output = model(each_x)

            if i == 0:
                loss = F.cross_entropy(output[0], label)  # logit
            else:
                loss = loss + F.cross_entropy(output[0], label)  # logit
            if using_aux_logit:
                loss = loss + F.cross_entropy(output[1], label)  # aux_logit
        loss.backward()
        grad = noise.grad.data



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

    assert rounddown(adv.max().item()) <= 1 and rounddown(adv.min().item()) >= 0
    assert rounddown((adv - img).min().item()) >= -eps and rounddown((adv - img).max()) <= eps

    if opt.visualize:
        for index in range(opt.batch_size):
            import time
            with torch.no_grad():
                time_start = time.time()
                output = model(torch.unsqueeze(img[index], 0))
                time_2 = time.time()
                print(f"classify the origin image:{time_2 - time_start}")
                cam = generate_cam(model, module_name, features_blobs, torch.unsqueeze(img[index], 0),
                                   target_class=torch.argmax(output[0]), device=device)
                time_3 = time.time()

                print(f"generate the cmp for original image{time_3 - time_2}")
                save_class_activation_images(img[index], cam, f'benign_{use_model}_{index}_{torch.argmax(output[0])}',
                                             dictionary='admix_TI_FGSM')
                time_4 = time.time()
                adversarial = model(torch.unsqueeze(adv[index], 0))
                time_5 = time.time()
                print(f"classify the adv mage:{time_5 - time_4}")
                cam = generate_cam(model, module_name, features_blobs, torch.unsqueeze(adv[index], 0),
                                   target_class=torch.argmax(adversarial[0]), device=device)
                time_6 = time.time()
                print(f"generate the cmp for adv image{time_6 - time_5}")
                save_class_activation_images(adv[index], cam, f'{use_model}_{index}_{torch.argmax(adversarial[0])}',
                                             dictionary='admix_TI_FGSM')

    return adv







def batch_grad(model, img, label,using_aux_logit,noise,grad):
    for iter  in range(20):
        neighbor = torch.cuda.FloatTensor(img.size())
        torch.randn(img.size(), out=neighbor)
        img2 =img + neighbor *1.5
        x_neighbor = img2.clone() +noise
        x_neighbor_2 = 1/2. * x_neighbor
        x_neighbor_4 = 1/4. * x_neighbor
        x_neighbor_8 = 1/8. * x_neighbor
        x_neighbor_16 = 1/16. * x_neighbor

        for i, each_x in enumerate([x_neighbor, x_neighbor_2, x_neighbor_4, x_neighbor_8,x_neighbor_16]):
            zero_gradients(noise)
            output = model(each_x)
            loss = F.cross_entropy(output[0], label)  # logit
            if using_aux_logit:
                loss += F.cross_entropy(output[1], label)  # aux_logit
            loss.backward()

            grad +=noise.grad.data*(1/2)**i

    return  grad


def gkern(kernlen=21, nsig=3):
  """Returns a 2D Gaussian kernel array."""
  import scipy.stats as st

  x = np.linspace(-nsig, nsig, kernlen)
  kern1d = st.norm.pdf(x)
  kernel_raw = np.outer(kern1d, kern1d)
  kernel = kernel_raw / kernel_raw.sum()
  return kernel
def TranslationInvariantAttack(model, img, label,using_aux_logit,use_diversity=True):
    X_pert = img.clone()
    batch,channel,H,W=X_pert.shape
    kernel = gkern(15, 3).astype(np.float32)
    stack_kernel = np.stack([kernel, kernel, kernel])
    stack_kernel = np.expand_dims(stack_kernel, 0)
    stack_kernel = torch.tensor(stack_kernel).cuda()
    stack_kernel=stack_kernel.repeat(batch,1,1,1)
    iteration,__,H_Kernel,W_Kernel=stack_kernel.shape
    stack_kernel= stack_kernel.transpose(0,1)
    stack_kernel=stack_kernel.reshape([batch*channel,1,H_Kernel,W_Kernel])

    eps = opt.max_epsilon / 255.0
    num_iter = opt.num_iter
    alpha = eps / num_iter
    momentum = opt.momentum

    noise = torch.zeros_like(img, requires_grad=True)
    X_pert = img.clone()
    # X_pert.requires_grad = True
    old_grad = 0.0
    for i in range(num_iter):
        zero_gradients(noise)

        if use_diversity:
            x =input_diversity(X_pert+noise, p=0.5, image_width=299, image_resize=330)
        else:
            x=X_pert+ noise
        output = model(x)
        loss = F.cross_entropy(output[0], label)  # logit
        if using_aux_logit:
            loss += F.cross_entropy(output[1], label)  # aux_logit
        loss.backward()
        grad = noise.grad.data
        #translation invariant
        grad =grad.reshape([1, batch * channel, H, W])
        grad=nn.functional.conv2d(grad,stack_kernel,padding='same',groups=channel*batch)
        grad= grad.reshape([batch,  channel, H, W])
        # momentum
        grad = grad / torch.abs(grad).sum([1, 2, 3], keepdim=True)
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
    assert rounddown(adv.max().item())<=1 and rounddown(adv.min().item())>=0
    assert rounddown((adv - img).min().item())>= -eps and rounddown((adv - img).max())<=eps

    if opt.visualize:
        for index in range(opt.batch_size):
            import time
            with torch.no_grad():
                time_start = time.time()
                output = model(torch.unsqueeze(img[index],0))
                time_2 = time.time()
                print(f"classify the origin image:{time_2-time_start}")
                cam = generate_cam(model, module_name, features_blobs, torch.unsqueeze(img[index],0), target_class=torch.argmax(output[0]),device=device)
                time_3 = time.time()

                print(f"generate the cmp for original image{time_3-time_2}")
                save_class_activation_images(img[index], cam, f'benign_{use_model}_{index}_{torch.argmax(output[0])}', dictionary='TIattack')
                time_4 = time.time()
                adversarial = model(torch.unsqueeze(adv[index],0))
                time_5= time.time()
                print(f"classify the adv mage:{time_5-time_4}")
                cam = generate_cam(model, module_name, features_blobs, torch.unsqueeze(adv[index],0), target_class=torch.argmax(adversarial[0]),device=device)
                time_6= time.time()
                print(f"generate the cmp for adv image{time_6-time_5}")
                save_class_activation_images(adv[index], cam, f'{use_model}_{index}_{torch.argmax(adversarial[0])}', dictionary='TIattack')


    return adv

import numpy as np

import scipy.stats as st


def project_noise(x, stack_kern, kern_size,channel,batch,H,W):
    x = torch.nn.functional.pad(x, (kern_size,kern_size,kern_size,kern_size,0,0,0,0), "constant", 0)
    x = x.reshape([1, batch * channel, H+2*kern_size, W+2*kern_size])


    x = nn.functional.conv2d(x,stack_kern,padding='valid',groups=channel*batch)
    x=x.reshape([batch, channel, H, W])
    return x
def project_kern(kern_size,batch=10):

    kern = np.ones((kern_size, kern_size), dtype=np.float32) / (kern_size ** 2 - 1)
    kern[kern_size // 2, kern_size // 2] = 0.0
    kern = kern.astype(np.float32)
    stack_kern = np.stack([kern, kern, kern])
    channel=3
    stack_kern = np.expand_dims(stack_kern, 0)
    stack_kern = torch.tensor(stack_kern).cuda()
    stack_kern=stack_kern.repeat(batch,1,1,1)
    iteration, __, H_Kernel, W_Kernel = stack_kern.shape
    stack_kern= stack_kern.transpose(0,1)
    stack_kern=stack_kern.reshape([batch*channel,1,H_Kernel,W_Kernel])
    return stack_kern, kern_size // 2
def admix(x,size=3):
    portion=0.2
    # size=3 #mixup
    return  torch.cat(tuple([(x + portion * x[torch.randperm(x.size(0))]) for _ in range(size)]), axis=0)/(1+portion*size)




def channel_shuffle(x):
    batchsize, num_channels, height, width = x.data.size()

    x2=x
    x3=x
    x4=x
    for i in range(x.shape[0]):
        tem=x2[i][1]
        x2[i][1]=x2[i][2]
        x2[i][2]=tem
    return x2


def channel_shuffle_TI_FGSM(model, img, label, using_aux_logit):

    eps = opt.max_epsilon / 255.0
    # channel_shuffle = torch.nn.ChannelShuffle(groups=3)
    num_iter = opt.num_iter
    alpha = eps / num_iter
    momentum = 1# set in the original paper
    grad=0
    X_pert  = img.clone()
    noise = torch.zeros_like(img, requires_grad=True)
    old_grad=torch.zeros_like(img)
    # label = torch.cat(tuple([label] * 3))


    batch,channel,H,W=X_pert.shape
    kernel = gkern(7, 3).astype(np.float32)
    stack_kernel = np.stack([kernel, kernel, kernel])
    stack_kernel = np.expand_dims(stack_kernel, 0)
    stack_kernel = torch.tensor(stack_kernel).cuda()
    stack_kernel=stack_kernel.repeat(batch,1,1,1)
    iteration,__,H_Kernel,W_Kernel=stack_kernel.shape
    stack_kernel= stack_kernel.transpose(0,1)
    stack_kernel=stack_kernel.reshape([batch*channel,1,H_Kernel,W_Kernel])


    for i in range(num_iter):
        zero_gradients(noise)
        # set_trace()
        x_cs=channel_shuffle((X_pert+ noise))
        # save_img(x_cs, [str(i)+".jpeg" for i in range(x_cs.shape[0])], opt.output_dir)
        x_origin=X_pert+ noise
        x_nes_2=1/2*x_origin
        x_nes_4 = 1 / 4 *x_origin
        x_nes_8 = 1 / 8 *x_origin
        x_nes_16 = 1 / 16 * x_origin
        x_nes_cs_2=1/2*x_cs
        x_nes_cs_4 = 1 / 4 * x_cs
        x_nes_cs_8 = 1 / 8 * x_cs
        x_nes_cs_16 = 1 / 16 * x_cs
        for i,each_x in enumerate([x_origin,x_nes_2 ,x_nes_4,x_nes_8,x_nes_16,x_cs,x_nes_cs_2 ,x_nes_cs_4,x_nes_cs_8,x_nes_cs_16]):
            output = model(input_diversity(each_x))
            # set_trace()
            if i ==0:
                loss = F.cross_entropy(output[0], label)  # logit
            else:
                loss=loss+ F.cross_entropy(output[0], label)  # logit
            if using_aux_logit:
                loss =loss + F.cross_entropy(output[1], label)  # aux_logit
        loss.backward()
        grad = noise.grad.data


        #translation invariant
        grad =grad.reshape([1, batch * channel, H, W])
        grad=nn.functional.conv2d(grad,stack_kernel,padding='same',groups=channel*batch)
        grad= grad.reshape([batch,  channel, H, W])

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
    assert rounddown(adv.max().item())<=1 and rounddown(adv.min().item())>=0
    assert rounddown((adv - img).min().item())>= -eps and rounddown((adv - img).max())<=eps


    if opt.visualize:
        for index in range(opt.batch_size):
            import time
            with torch.no_grad():
                time_start = time.time()
                output = model(torch.unsqueeze(img[index],0))
                time_2 = time.time()
                print(f"classify the origin image:{time_2-time_start}")
                cam = generate_cam(model, module_name, features_blobs, torch.unsqueeze(img[index],0), target_class=torch.argmax(output[0]),device=device)
                time_3 = time.time()

                print(f"generate the cmp for original image{time_3-time_2}")
                save_class_activation_images(img[index], cam, f'benign_{use_model}_{index}_{torch.argmax(output[0])}', dictionary='admix_FGSM')
                time_4 = time.time()
                adversarial = model(torch.unsqueeze(adv[index],0))
                time_5= time.time()
                print(f"classify the adv mage:{time_5-time_4}")
                cam = generate_cam(model, module_name, features_blobs, torch.unsqueeze(adv[index],0), target_class=torch.argmax(adversarial[0]),device=device)
                time_6= time.time()
                print(f"generate the cmp for adv image{time_6-time_5}")
                save_class_activation_images(adv[index], cam, f'{use_model}_{index}_{torch.argmax(adversarial[0])}', dictionary='admix_FGSM')



    return adv

def rgb_to_grayscale(img, num_output_channels: int = 1,r_c=0.2989,g_c=0.587,b_c=0.114) :
    if img.ndim < 3:
        raise TypeError(f"Input image tensor should have at least 3 dimensions, but found {img.ndim}")

    if num_output_channels not in (1, 3):
        raise ValueError("num_output_channels should be either 1 or 3")

    r, g, b = img.unbind(dim=-3)
    # This implementation closely follows the TF one:
    # https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/ops/image_ops_impl.py#L2105-L2138
    l_img = (r_c * r + g_c* g + b_c* b).to(img.dtype)
    l_img = l_img.unsqueeze(dim=-3)

    if num_output_channels == 3:
        return l_img.expand(img.shape)

    return l_img

def greyschale_TI_FGSM(model, img, label, using_aux_logit):
    eps = opt.max_epsilon / 255.0
    # channel_shuffle = torch.nn.ChannelShuffle(groups=3)
    num_iter = opt.num_iter
    alpha = eps / num_iter
    momentum = 1# set in the original paper
    grad=0
    X_pert  = img.clone()
    noise = torch.zeros_like(img, requires_grad=True)
    old_grad=torch.zeros_like(img)
    # label = torch.cat(tuple([label] * 3))


    batch,channel,H,W=X_pert.shape
    kernel = gkern(7, 3).astype(np.float32)
    stack_kernel = np.stack([kernel, kernel, kernel])
    stack_kernel = np.expand_dims(stack_kernel, 0)
    stack_kernel = torch.tensor(stack_kernel).cuda()
    stack_kernel=stack_kernel.repeat(batch,1,1,1)
    iteration,__,H_Kernel,W_Kernel=stack_kernel.shape
    stack_kernel= stack_kernel.transpose(0,1)
    stack_kernel=stack_kernel.reshape([batch*channel,1,H_Kernel,W_Kernel])


    for i in range(num_iter):
        zero_gradients(noise)
        # set_trace()
        # x_cs=channel_shuffle((X_pert+ noise))
        # save_img(x_cs, [str(i)+".jpeg" for i in range(x_cs.shape[0])], opt.output_dir)
        transform = transforms.Grayscale(num_output_channels=3)
        x_grey=transform(X_pert+ noise)
        x_origin=X_pert+ noise
        x_nes_2=1/2*x_origin
        x_nes_4 = 1 / 4 *x_origin
        x_nes_8 = 1 / 8 *x_origin
        x_nes_16 = 1 / 16 * x_origin
        x_nes_cs_2=1/2*x_grey
        x_nes_cs_4 = 1 / 4 * x_grey
        x_nes_cs_8 = 1 / 8 * x_grey
        x_nes_cs_16 = 1 / 16 * x_grey
        for i,each_x in enumerate([x_origin,x_nes_2 ,x_nes_4,x_nes_8,x_nes_16,x_grey,x_nes_cs_2 ,x_nes_cs_4,x_nes_cs_8,x_nes_cs_16]):
            output = model(input_diversity(each_x))
            # set_trace()
            if i ==0:
                loss = F.cross_entropy(output[0], label)  # logit
            else:
                loss=loss+ F.cross_entropy(output[0], label)  # logit
            if using_aux_logit:
                loss =loss + F.cross_entropy(output[1], label)  # aux_logit
        loss.backward()
        grad = noise.grad.data


        #translation invariant
        grad =grad.reshape([1, batch * channel, H, W])
        grad=nn.functional.conv2d(grad,stack_kernel,padding='same',groups=channel*batch)
        grad= grad.reshape([batch,  channel, H, W])

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
    assert rounddown(adv.max().item())<=1 and rounddown(adv.min().item())>=0
    assert rounddown((adv - img).min().item())>= -eps and rounddown((adv - img).max())<=eps


    if opt.visualize:
        for index in range(opt.batch_size):
            import time
            with torch.no_grad():
                time_start = time.time()
                output = model(torch.unsqueeze(img[index],0))
                time_2 = time.time()
                print(f"classify the origin image:{time_2-time_start}")
                cam = generate_cam(model, module_name, features_blobs, torch.unsqueeze(img[index],0), target_class=torch.argmax(output[0]),device=device)
                time_3 = time.time()

                print(f"generate the cmp for original image{time_3-time_2}")
                save_class_activation_images(img[index], cam, f'benign_{use_model}_{index}_{torch.argmax(output[0])}', dictionary='greyschale_TI_FGSM')
                time_4 = time.time()
                adversarial = model(torch.unsqueeze(adv[index],0))
                time_5= time.time()
                print(f"classify the adv mage:{time_5-time_4}")
                cam = generate_cam(model, module_name, features_blobs, torch.unsqueeze(adv[index],0), target_class=torch.argmax(adversarial[0]),device=device)
                time_6= time.time()
                print(f"generate the cmp for adv image{time_6-time_5}")
                save_class_activation_images(adv[index], cam, f'{use_model}_{index}_{torch.argmax(adversarial[0])}', dictionary='greyschale_TI_FGSM')



    return adv

def BCSH_TI_FGSM(model, img, label, using_aux_logit):
    eps = opt.max_epsilon / 255.0
    # channel_shuffle = torch.nn.ChannelShuffle(groups=3)
    num_iter = opt.num_iter
    alpha = eps / num_iter
    momentum = 1# set in the original paper
    grad=0
    X_pert  = img.clone()
    noise = torch.zeros_like(img, requires_grad=True)
    old_grad=torch.zeros_like(img)
    # label = torch.cat(tuple([label] * 3))


    batch,channel,H,W=X_pert.shape
    kernel = gkern(7, 3).astype(np.float32)
    stack_kernel = np.stack([kernel, kernel, kernel])
    stack_kernel = np.expand_dims(stack_kernel, 0)
    stack_kernel = torch.tensor(stack_kernel).cuda()
    stack_kernel=stack_kernel.repeat(batch,1,1,1)
    iteration,__,H_Kernel,W_Kernel=stack_kernel.shape
    stack_kernel= stack_kernel.transpose(0,1)
    stack_kernel=stack_kernel.reshape([batch*channel,1,H_Kernel,W_Kernel])


    for i in range(num_iter):
        zero_gradients(noise)

        # x_cs=channel_shuffle((X_pert+ noise))
        # save_img(x_cs, [str(i)+".jpeg" for i in range(x_cs.shape[0])], opt.output_dir)
        transform = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5)
        x_bcsh=transform(X_pert+ noise)
        x_origin=X_pert+ noise
        x_nes_2=1/2*x_origin
        x_nes_4 = 1 / 4 *x_origin
        x_nes_8 = 1 / 8 *x_origin
        x_nes_16 = 1 / 16 * x_origin
        x_nes_cs_2=1/2*x_bcsh
        x_nes_cs_4 = 1 / 4 * x_bcsh
        x_nes_cs_8 = 1 / 8 * x_bcsh
        x_nes_cs_16 = 1 / 16 * x_bcsh
        for i,each_x in enumerate([x_origin,x_nes_2 ,x_nes_4,x_nes_8,x_nes_16,x_bcsh,x_nes_cs_2 ,x_nes_cs_4,x_nes_cs_8,x_nes_cs_16]):
            output = model(input_diversity(each_x))

            if i ==0:
                loss = F.cross_entropy(output[0], label)  # logit
            else:
                loss=loss+ F.cross_entropy(output[0], label)  # logit
            if using_aux_logit:
                loss =loss + F.cross_entropy(output[1], label)  # aux_logit
        loss.backward()
        grad = noise.grad.data


        #translation invariant
        grad =grad.reshape([1, batch * channel, H, W])
        grad=nn.functional.conv2d(grad,stack_kernel,padding='same',groups=channel*batch)
        grad= grad.reshape([batch,  channel, H, W])

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
    assert rounddown(adv.max().item())<=1 and rounddown(adv.min().item())>=0
    assert rounddown((adv - img).min().item())>= -eps and rounddown((adv - img).max())<=eps


    if opt.visualize:
        for index in range(opt.batch_size):
            import time
            with torch.no_grad():
                time_start = time.time()
                output = model(torch.unsqueeze(img[index],0))
                time_2 = time.time()
                print(f"classify the origin image:{time_2-time_start}")
                cam = generate_cam(model, module_name, features_blobs, torch.unsqueeze(img[index],0), target_class=torch.argmax(output[0]),device=device)
                time_3 = time.time()

                print(f"generate the cmp for original image{time_3-time_2}")
                save_class_activation_images(img[index], cam, f'benign_{use_model}_{index}_{torch.argmax(output[0])}', dictionary='BCSH_TI_FGSM')
                time_4 = time.time()
                adversarial = model(torch.unsqueeze(adv[index],0))
                time_5= time.time()
                print(f"classify the adv mage:{time_5-time_4}")
                cam = generate_cam(model, module_name, features_blobs, torch.unsqueeze(adv[index],0), target_class=torch.argmax(adversarial[0]),device=device)
                time_6= time.time()
                print(f"generate the cmp for adv image{time_6-time_5}")
                save_class_activation_images(adv[index], cam, f'{use_model}_{index}_{torch.argmax(adversarial[0])}', dictionary='BCSH_TI_FGSM')



    return adv

def mycutout(img,p=0.5, scale=(0.02, 0.4), ratio=(0.4, 1 / 0.4), value=(0, 255), pixel_level=False, inplace=False):
    if random.random() < p:
    # if True:

        batch, img_c,img_h, img_w = img.shape
        s = random.uniform(*scale)
        s = s * img_h * img_w
        r = random.uniform(*ratio)
        w = int(math.sqrt(s / r))
        h = int(math.sqrt(s * r))
        left = random.randint(0, img_w - w)
        top = random.randint(0, img_h - h)
        c = torch.tensor(0).to('cuda')

        for i in range(batch):
            img[i,:,left:left + w,top:top + h]=c
        # save_img(img, [str(i) + "cutout.jpeg" for i in range(img.shape[0])], opt.output_dir)
        return  img
    else:
        return  img


def utimate_admixFGSM(model, img,img_n, label, using_aux_logit,order):

    order=str(bin(order))[2:]
    extrazero=6-len(order)
    order=extrazero*'0'+order
    eps = opt.max_epsilon / 255.0
    # channel_shuffle = torch.nn.ChannelShuffle(groups=3)
    num_iter = opt.num_iter
    alpha = eps / num_iter
    momentum = 1  # set in the original paper
    grad = 0
    X_pert = img.clone()
    noise = torch.zeros_like(img, requires_grad=True)
    old_grad = torch.zeros_like(img)
    # label = torch.cat(tuple([label] * 3))

    batch, channel, H, W = X_pert.shape
    kernel = gkern(7, 3).astype(np.float32)
    stack_kernel = np.stack([kernel, kernel, kernel])
    stack_kernel = np.expand_dims(stack_kernel, 0)
    stack_kernel = torch.tensor(stack_kernel).cuda()
    stack_kernel = stack_kernel.repeat(batch, 1, 1, 1)
    iteration, __, H_Kernel, W_Kernel = stack_kernel.shape
    stack_kernel = stack_kernel.transpose(0, 1)
    stack_kernel = stack_kernel.reshape([batch * channel, 1, H_Kernel, W_Kernel])

    for i in range(num_iter):
        zero_gradients(noise)
        # set_trace()
        # x_cs=channel_shuffle((X_pert+ noise))
        # save_img(x_cs, [str(i)+".jpeg" for i in range(x_cs.shape[0])], opt.output_dir)


        augmented=[]
        augmented_SI=[]
        augmented.append(X_pert + noise)
        for index,whethertouse in enumerate(order):
            if whethertouse=='1':
                if index==0:
                    transform = transforms.Grayscale(num_output_channels=3)
                    x_grey = transform(X_pert + noise)
                    augmented.append(x_grey)

                elif index==1:
                    # rand_index = torch.randperm(img.shape[0]).cuda()
                    # bbx1, bby1, bbx2, bby2 = rand_bbox(img.shape)
                    # x = X_pert + noise
                    #
                    # x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
                    augmented.append(mycutout(X_pert + noise, p=0.5, ratio=(1, 1), value=(0, 1)))
                elif index==2:
                    x_neural = img_n.clone()
                    augmented.append(x_neural+noise)
                elif index==3:
                    augmented.append(Edge_Enhance(X_pert + noise))
                elif index==4:
                    transform = transforms.Compose([
                        ImageNetPolicy,
                        transforms.ToTensor()
                    ])

                    pil_trans = ToPILImage()
                    x_RE1 = torch.zeros_like(img)
                    for i in range(img.shape[0]):
                        x_RE1[i] = transform(pil_trans(X_pert[i])).cuda() + noise[i]
                    augmented.append(x_RE1)

        augmented.append(admix(X_pert + noise,1))
        if order[5] == '1':
            for item in augmented:
                augmented_SI.append(item/2)
                augmented_SI.append(item / 4)
                augmented_SI.append(item / 8)
                augmented_SI.append(item /16)






        for i, each_x in enumerate(augmented_SI+augmented):
            # x_nes_RE2_2,x_nes_RE2_4,x_nes_RE2_8,x_nes_RE2_16,x_nes_RE3_2,x_nes_RE3_4,x_nes_RE3_8,x_nes_RE3_16]):
            if order[5]=='1':

                output = model(input_diversity(each_x))
            else:
                output = model(each_x)
            # set_trace()
            if i == 0:
                loss = F.cross_entropy(output[0], label)  # logit
            else:
                loss = loss + F.cross_entropy(output[0], label)  # logit
            if using_aux_logit:
                loss = loss + F.cross_entropy(output[1], label)  # aux_logit
        loss.backward()
        grad = noise.grad.data
        if order[5] == '1':
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
def bestcombo_admixFGSMkernel(model, img,label, using_aux_logit,kernel_size):

    eps = opt.max_epsilon / 255.0
    # channel_shuffle = torch.nn.ChannelShuffle(groups=3)
    num_iter = opt.num_iter
    alpha = eps / num_iter
    momentum = 1  # set in the original paper
    grad = 0
    X_pert = img.clone()
    noise = torch.zeros_like(img, requires_grad=True)
    old_grad = torch.zeros_like(img)
    # label = torch.cat(tuple([label] * 3))

    batch, channel, H, W = X_pert.shape
    kernel = gkern(kernel_size, 3).astype(np.float32)
    stack_kernel = np.stack([kernel, kernel, kernel])
    stack_kernel = np.expand_dims(stack_kernel, 0)
    stack_kernel = torch.tensor(stack_kernel).cuda()
    stack_kernel = stack_kernel.repeat(batch, 1, 1, 1)
    iteration, __, H_Kernel, W_Kernel = stack_kernel.shape
    stack_kernel = stack_kernel.transpose(0, 1)
    stack_kernel = stack_kernel.reshape([batch * channel, 1, H_Kernel, W_Kernel])

    for i in range(num_iter):
        zero_gradients(noise)
        # set_trace()
        # x_cs=channel_shuffle((X_pert+ noise))
        # save_img(x_cs, [str(i)+".jpeg" for i in range(x_cs.shape[0])], opt.output_dir)


        augmented=[]
        augmented_SI=[]
        augmented.append(X_pert + noise)



        transform = transforms.Grayscale(num_output_channels=3)
        x_grey = transform(X_pert + noise)
        augmented.append(x_grey)

        augmented.append(mycutout(X_pert + noise, p=0.5, ratio=(1, 1), value=(0, 1)))


        augmented.append(Edge_Enhance(X_pert + noise))

        transform = transforms.Compose([
            ImageNetPolicy,
            transforms.ToTensor()
        ])

        pil_trans = ToPILImage()
        x_RE1 = torch.zeros_like(img)
        for i in range(img.shape[0]):
            x_RE1[i] = transform(pil_trans(X_pert[i])).cuda() + noise[i]
        augmented.append(x_RE1)
        augmented.append(admix(X_pert + noise,1))

        for item in augmented:
            augmented_SI.append(item/2)
            augmented_SI.append(item / 4)
            augmented_SI.append(item / 8)
            augmented_SI.append(item /16)






        for i, each_x in enumerate(augmented_SI+augmented):
            output = model(input_diversity(each_x))

            # set_trace()
            if i == 0:
                loss = F.cross_entropy(output[0], label)  # logit
            else:
                loss = loss + F.cross_entropy(output[0], label)  # logit
            if using_aux_logit:
                loss = loss + F.cross_entropy(output[1], label)  # aux_logit
        loss.backward()
        grad = noise.grad.data

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
def colorjitter_TI_FGSM(model, img, label, using_aux_logit):
    eps = opt.max_epsilon / 255.0
    # channel_shuffle = torch.nn.ChannelShuffle(groups=3)
    num_iter = opt.num_iter
    alpha = eps / num_iter
    momentum = 1# set in the original paper
    grad=0
    X_pert  = img.clone()
    noise = torch.zeros_like(img, requires_grad=True)
    old_grad=torch.zeros_like(img)
    # label = torch.cat(tuple([label] * 3))


    batch,channel,H,W=X_pert.shape
    kernel = gkern(7, 3).astype(np.float32)
    stack_kernel = np.stack([kernel, kernel, kernel])
    stack_kernel = np.expand_dims(stack_kernel, 0)
    stack_kernel = torch.tensor(stack_kernel).cuda()
    stack_kernel=stack_kernel.repeat(batch,1,1,1)
    iteration,__,H_Kernel,W_Kernel=stack_kernel.shape
    stack_kernel= stack_kernel.transpose(0,1)
    stack_kernel=stack_kernel.reshape([batch*channel,1,H_Kernel,W_Kernel])


    for i in range(num_iter):
        zero_gradients(noise)
        # set_trace()

        # x_cs=channel_shuffle((X_pert+ noise))
        # save_img(x_cs, [str(i)+".jpeg" for i in range(x_cs.shape[0])], opt.output_dir)
        x1,x2,x3=group_pca_color_augmention((X_pert+ noise))

        x_origin=X_pert+ noise
        x_nes_2=1/2*x_origin
        x_nes_4 = 1 / 4 *x_origin
        x_nes_8 = 1 / 8 *x_origin
        x_nes_16 = 1 / 16 * x_origin
        x_nes_cs_2=1/2*x1
        x_nes_cs_4 = 1 / 4 * x1
        x_nes_cs_8 = 1 / 8 * x1
        x_nes_cs_16 = 1 / 16 * x1
        x_nes_pca_2=1/2*x2
        x_nes_pca_4 = 1 / 4 * x2
        x_nes_pca_8 = 1 / 8 * x2
        x_nes_pca_16 = 1 / 16 * x2
        for i,each_x in enumerate([x_origin,x1,x2,x_nes_2 ,x_nes_4,x_nes_8,x_nes_16,x_nes_cs_2 ,x_nes_cs_4,x_nes_cs_8,x_nes_cs_16,x_nes_pca_2,x_nes_pca_4,x_nes_pca_8,x_nes_pca_16]):
            output = model(input_diversity(each_x))
            # set_trace()
            if i ==0:
                loss = F.cross_entropy(output[0], label)  # logit
            else:
                loss=loss+ F.cross_entropy(output[0], label)  # logit
            if using_aux_logit:
                loss =loss + F.cross_entropy(output[1], label)  # aux_logit
        loss.backward()
        grad = noise.grad.data


        #translation invariant
        grad =grad.reshape([1, batch * channel, H, W])
        grad=nn.functional.conv2d(grad,stack_kernel,padding='same',groups=channel*batch)
        grad= grad.reshape([batch,  channel, H, W])

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
    assert rounddown(adv.max().item())<=1 and rounddown(adv.min().item())>=0
    assert rounddown((adv - img).min().item())>= -eps and rounddown((adv - img).max())<=eps


    if opt.visualize:
        for index in range(opt.batch_size):
            import time
            with torch.no_grad():
                time_start = time.time()
                output = model(torch.unsqueeze(img[index],0))
                time_2 = time.time()
                print(f"classify the origin image:{time_2-time_start}")
                cam = generate_cam(model, module_name, features_blobs, torch.unsqueeze(img[index],0), target_class=torch.argmax(output[0]),device=device)
                time_3 = time.time()

                print(f"generate the cmp for original image{time_3-time_2}")
                save_class_activation_images(img[index], cam, f'benign_{use_model}_{index}_{torch.argmax(output[0])}', dictionary='colorjitter_TI_FGSM')
                time_4 = time.time()
                adversarial = model(torch.unsqueeze(adv[index],0))
                time_5= time.time()
                print(f"classify the adv mage:{time_5-time_4}")
                cam = generate_cam(model, module_name, features_blobs, torch.unsqueeze(adv[index],0), target_class=torch.argmax(adversarial[0]),device=device)
                time_6= time.time()
                print(f"generate the cmp for adv image{time_6-time_5}")
                save_class_activation_images(adv[index], cam, f'{use_model}_{index}_{torch.argmax(adversarial[0])}', dictionary='colorjitter_TI_FGSM')



    return adv
# the use in Zebin's code in for ultimate_combo:
# utimate_admixFGSM(model, img, images_n, label, using_aux_logit, 55)
def ultimate_admixFGSM(model, img, label, using_aux_logit,cfg):
    order = cfg['order']
    img_n = None
    order=str(bin(order))[2:]
    extrazero=6-len(order)
    order=extrazero*'0'+order
    eps = cfg['epsilon'] # AW
    # channel_shuffle = torch.nn.ChannelShuffle(groups=3)
    num_iter = cfg['niters'] #AW
    alpha = eps / num_iter
    momentum = 1  # set in the original paper
    grad = 0
    X_pert = img.clone()
    noise = torch.zeros_like(img, requires_grad=True)
    old_grad = torch.zeros_like(img)
    # label = torch.cat(tuple([label] * 3))

    batch, channel, H, W = X_pert.shape
    kernel = gkern(7, 3).astype(np.float32)
    stack_kernel = np.stack([kernel]*channel) # AW: support different channel numbers
    # stack_kernel = np.stack([kernel, kernel, kernel])
    stack_kernel = np.expand_dims(stack_kernel, 0)
    stack_kernel = torch.tensor(stack_kernel).cuda()
    stack_kernel = stack_kernel.repeat(batch, 1, 1, 1)
    iteration, __, H_Kernel, W_Kernel = stack_kernel.shape
    stack_kernel = stack_kernel.transpose(0, 1)
    stack_kernel = stack_kernel.reshape([batch * channel, 1, H_Kernel, W_Kernel])

    for i in range(num_iter):
        zero_gradients(noise)
        # set_trace()
        # x_cs=channel_shuffle((X_pert+ noise))
        # save_img(x_cs, [str(i)+".jpeg" for i in range(x_cs.shape[0])], opt.output_dir)


        augmented=[]
        augmented_SI=[]
        augmented.append(X_pert + noise)
        for index,whethertouse in enumerate(order):
            if whethertouse=='1':
                if index==0:
                    if channel==3:
                        transform = transforms.Grayscale(num_output_channels=3)
                        x_grey = transform(X_pert + noise)
                    else:
                        assert(channel==1)
                        x_grey = X_pert + noise # no point in greyscaling a grey img
                    augmented.append(x_grey)
                elif index==1:
                    # rand_index = torch.randperm(img.shape[0]).cuda()
                    # bbx1, bby1, bbx2, bby2 = rand_bbox(img.shape)
                    # x = X_pert + noise
                    #
                    # x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
                    augmented.append(mycutout(X_pert + noise, p=0.5, ratio=(1, 1), value=(0, 1)))
                elif index==2:
                    x_neural = img_n.clone()
                    augmented.append(x_neural+noise)
                elif index==3:
                    augmented.append(Edge_Enhance(X_pert + noise))
                elif index==4:
                    transform = transforms.Compose([
                        ImageNetPolicy,
                        transforms.ToTensor()
                    ])

                    pil_trans = ToPILImage()
                    x_RE1 = torch.zeros_like(img)
                    for i in range(img.shape[0]):
                        x_RE1[i] = transform(pil_trans(X_pert[i])).cuda() + noise[i]
                    augmented.append(x_RE1)

        augmented.append(admix(X_pert + noise,1))
        if order[5] == '1':
            for item in augmented:
                augmented_SI.append(item/2)
                augmented_SI.append(item / 4)
                augmented_SI.append(item / 8)
                augmented_SI.append(item /16)






        for i, each_x in enumerate(augmented_SI+augmented):
            # x_nes_RE2_2,x_nes_RE2_4,x_nes_RE2_8,x_nes_RE2_16,x_nes_RE3_2,x_nes_RE3_4,x_nes_RE3_8,x_nes_RE3_16]):
            if order[5]=='1':

                output = model(input_diversity(each_x))
            else:
                output = model(each_x)
            # set_trace()
            if i == 0:
                loss = F.cross_entropy(output, label)  # logit
            else:
                loss = loss + F.cross_entropy(output, label)  # logit
            if using_aux_logit:
                loss = loss + F.cross_entropy(output[1], label)  # aux_logit # AW: should be false
        loss.backward()
        grad = noise.grad.data
        if order[5] == '1':
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
def SI_DI_TI_MIFGSM(model, img, label, using_aux_logit):
    eps = opt.max_epsilon / 255.0
    num_iter = opt.num_iter
    alpha = eps / num_iter
    momentum = 1# set in the original paper
    grad=0
    old_grad=0.0
    X_pert  = img.clone()

    batch,channel,H,W=X_pert.shape
    kernel = gkern(7, 3).astype(np.float32)
    stack_kernel = np.stack([kernel, kernel, kernel])
    stack_kernel = np.expand_dims(stack_kernel, 0)
    stack_kernel = torch.tensor(stack_kernel).cuda()
    stack_kernel=stack_kernel.repeat(batch,1,1,1)
    iteration,__,H_Kernel,W_Kernel=stack_kernel.shape
    stack_kernel= stack_kernel.transpose(0,1)
    stack_kernel=stack_kernel.reshape([batch*channel,1,H_Kernel,W_Kernel])


    noise = torch.zeros_like(img, requires_grad=True)

    for i in range(num_iter):
        zero_gradients(noise)
        x=X_pert+ noise
        x_nes_2=1/2*x
        x_nes_4 = 1 / 4 * x
        x_nes_8 = 1 / 8 * x
        x_nes_16 = 1 / 16 * x
        temp_grad=0
        for i,each_x in enumerate([x,x_nes_2 ,x_nes_4,x_nes_8,x_nes_16]):
            output = model(input_diversity(each_x))

            if i ==0:
                loss = F.cross_entropy(output[0], label)  # logit
            else:
                loss=loss+ F.cross_entropy(output[0], label)  # logit
            if using_aux_logit:
                loss =loss + F.cross_entropy(output[1], label)  # aux_logit
        loss.backward()
        grad = noise.grad.data

        # translation invariant
        grad =grad.reshape([1, batch * channel, H, W])
        grad=nn.functional.conv2d(grad,stack_kernel,padding='same',groups=channel*batch)
        grad= grad.reshape([batch,  channel, H, W])

        # MI-FGSM
        grad = grad / torch.abs(grad).sum([1,2,3], keepdim=True)
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
    assert rounddown(adv.max().item())<=1 and rounddown(adv.min().item())>=0
    assert rounddown((adv - img).min().item())>= -eps and rounddown((adv - img).max())<=eps





    return adv
def admix_TI_FGSM(model, img, label, using_aux_logit,cfg):
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
    if 'image' in cfg['dataset'] : #AW
        kernel = gkern(9, 3).astype(np.float32)
    elif cfg['dataset'] in ['mnist','cifar']: #AW
        kernel = gkern(3, 3).astype(np.float32)
    else:
        raise Exception(f'got dataset {cfg["dataset"]}')
    # stack_kernel = np.stack([kernel, kernel, kernel])
    stack_kernel = np.stack([kernel]*channel) # AW: support different channel numbers
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
def utimate_MIFGSM(model, img,img_n, label, using_aux_logit,order):

    order=str(bin(order))[2:]
    extrazero=6-len(order)
    order=extrazero*'0'+order
    eps = opt.max_epsilon / 255.0
    # channel_shuffle = torch.nn.ChannelShuffle(groups=3)
    num_iter = opt.num_iter
    alpha = eps / num_iter
    momentum = 1  # set in the original paper
    grad = 0
    X_pert = img.clone()
    noise = torch.zeros_like(img, requires_grad=True)
    old_grad = torch.zeros_like(img)
    # label = torch.cat(tuple([label] * 3))

    batch, channel, H, W = X_pert.shape
    kernel = gkern(7, 3).astype(np.float32)
    stack_kernel = np.stack([kernel, kernel, kernel])
    stack_kernel = np.expand_dims(stack_kernel, 0)
    stack_kernel = torch.tensor(stack_kernel).cuda()
    stack_kernel = stack_kernel.repeat(batch, 1, 1, 1)
    iteration, __, H_Kernel, W_Kernel = stack_kernel.shape
    stack_kernel = stack_kernel.transpose(0, 1)
    stack_kernel = stack_kernel.reshape([batch * channel, 1, H_Kernel, W_Kernel])

    for i in range(num_iter):
        zero_gradients(noise)
        # set_trace()
        # x_cs=channel_shuffle((X_pert+ noise))
        # save_img(x_cs, [str(i)+".jpeg" for i in range(x_cs.shape[0])], opt.output_dir)

        global randomseed
        randomseed += 1
        augmented=[]
        augmented_SI=[]
        augmented.append(X_pert + noise)
        seed_torch(randomseed)
        for index,whethertouse in enumerate(order):
            if whethertouse=='1':
                if index==0:
                    transform = transforms.Grayscale(num_output_channels=3)
                    x_grey = transform(X_pert + noise)
                    augmented.append(x_grey)

                elif index==1:
                    # rand_index = torch.randperm(img.shape[0]).cuda()
                    # bbx1, bby1, bbx2, bby2 = rand_bbox(img.shape)
                    # x = X_pert + noise
                    #
                    # x[:, :, bbx1:bbx2, bby1:bby2] = x[rand_index, :, bbx1:bbx2, bby1:bby2]
                    augmented.append(mycutout(X_pert + noise, p=0.5, ratio=(1, 1), value=(0, 1)))
                elif index==2:
                    x_neural = img_n.clone()
                    augmented.append(x_neural+noise)
                elif index==3:
                    augmented.append(Edge_Enhance(X_pert + noise))
                elif index==4:
                    transform = transforms.Compose([
                        ImageNetPolicy,
                        transforms.ToTensor()
                    ])

                    pil_trans = ToPILImage()
                    x_RE1 = torch.zeros_like(img)
                    for i in range(img.shape[0]):
                        x_RE1[i] = transform(pil_trans(X_pert[i])).cuda() + noise[i]
                    augmented.append(x_RE1)

        if order[5] == '1':
            for item in augmented:
                augmented_SI.append(item/2)
                augmented_SI.append(item / 4)
                augmented_SI.append(item / 8)
                augmented_SI.append(item /16)



        seed_torch(0)


        for i, each_x in enumerate(augmented_SI+augmented):
            # x_nes_RE2_2,x_nes_RE2_4,x_nes_RE2_8,x_nes_RE2_16,x_nes_RE3_2,x_nes_RE3_4,x_nes_RE3_8,x_nes_RE3_16]):
            if order[5]=='1':

                output = model(input_diversity(each_x))
            else:
                output = model(each_x)
            # set_trace()
            if i == 0:
                loss = F.cross_entropy(output[0], label)  # logit
            else:
                loss = loss + F.cross_entropy(output[0], label)  # logit
            if using_aux_logit:
                loss = loss + F.cross_entropy(output[1], label)  # aux_logit
        loss.backward()
        grad = noise.grad.data
        if order[5] == '1':
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

    if opt.visualize:
        for index in range(opt.batch_size):
            import time
            with torch.no_grad():
                time_start = time.time()
                output = model(torch.unsqueeze(img[index], 0))
                time_2 = time.time()
                print(f"classify the origin image:{time_2 - time_start}")
                cam = generate_cam(model, module_name, features_blobs, torch.unsqueeze(img[index], 0),
                                   target_class=torch.argmax(output[0]), device=device)
                time_3 = time.time()

                print(f"generate the cmp for original image{time_3 - time_2}")
                save_class_activation_images(img[index], cam, f'benign_{use_model}_{index}_{torch.argmax(output[0])}',
                                             dictionary='autoaugment_TI_FGSM')
                time_4 = time.time()
                adversarial = model(torch.unsqueeze(adv[index], 0))
                time_5 = time.time()
                print(f"classify the adv mage:{time_5 - time_4}")
                cam = generate_cam(model, module_name, features_blobs, torch.unsqueeze(adv[index], 0),
                                   target_class=torch.argmax(adversarial[0]), device=device)
                time_6 = time.time()
                print(f"generate the cmp for adv image{time_6 - time_5}")
                save_class_activation_images(adv[index], cam, f'{use_model}_{index}_{torch.argmax(adversarial[0])}',
                                             dictionary='autoaugment_TI_FGSM')

    return adv
global image_loss_record

global df

image_loss_record=0

def attack(model, img, label,model_name,benign=False,method='benign',images_n=None):
    """generate adversarial images"""

    using_aux_logit = not 'resnet' in model_name
    if benign:
        return img
    elif method=='SI_DI_TI_MIFGSM':
        return SI_DI_TI_MIFGSM(model, img, label, using_aux_logit)

    elif method=='channel_shuffle_TI_FGSM':
        return channel_shuffle_TI_FGSM(model, img, label, using_aux_logit)

    elif method=='greyscale_TI_FGSM':
        return greyschale_TI_FGSM(model, img, label,using_aux_logit)



    elif method=='fPCA_TI_FGSM':
        return colorjitter_TI_FGSM(model, img, label,using_aux_logit)#single_colorjitterMIFGSM(model, img, label, using_aux_logit)


    elif method=='colorjitter_TI_FGSM':
        return BCSH_TI_FGSM(model, img, label,using_aux_logit)



    elif method=='ultimate_combo' :


        return utimate_admixFGSM(model, img, images_n, label, using_aux_logit, 55)

    elif method=='admix_DI_TI_FGSM':
        return admix_TI_FGSM(model, img, label, using_aux_logit)
    elif method=='SI_DI_TI_MIFGSM':
        return  SI_DI_TI_MIFGSM(model, img, label, using_aux_logit)
    elif method.startswith('ultimateadmixFGSM'):
        order = int(method[17:])

        return utimate_admixFGSM(model, img, images_n, label, using_aux_logit, order)
    elif method.startswith('ultimateMIFGSM'):
        order=int(method[14:])

        return utimate_MIFGSM(model, img,images_n, label,using_aux_logit,order)

    elif method=='benign':
        return img







global module_name
global features_blobs
module_name = []
features_blobs=[]

# from scorecam_implement import *
def hook_feature(module, input, output):
    global module_name
    global features_blobs
    features_blobs.append(output[0].data.cpu().numpy())
    module_name.append(module)

def main():
    index=0
    transforms = T.Compose([T.ToTensor()])
    transforms2 = T.Compose([T.Resize(299), T.ToTensor()])

    df_res = pd.DataFrame(
        columns=['dst_benign_success',  'name'])
    # Load inputs
    inputs = ImageNet(opt.input_dir, opt.input_csv, transforms)
    neural = ImageNet('./neural-style-pt/output', opt.input_csv, transforms2)
    data_loader = DataLoader(inputs, batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=0)

    # transforms.Resize(224)
    data_loader_neural = DataLoader(neural, batch_size=opt.batch_size, shuffle=False, pin_memory=True, num_workers=0)

    input_num = len(inputs)

    # Create models
    models = get_models(list_nets, opt.model_dir)



    # Start iteration
    method=opt.method
    # print(method)
    surrogate_models = list_nets

    methods=[method]

    global use_model


    for skipnumber,surrogate in enumerate(surrogate_models):
        use_model=surrogate
        global df

        global image_loss_record
        image_loss_record = 0
        for method in methods:
            # Initialization parameters
            correct_num = {}
            logits = {}
            for net in list_nets:
                correct_num[net] = 0

            if opt.visualize:
                global module_name
                global features_blobs
                module_name = []
                features_blobs = []
                net_chilren = models[surrogate].children()
                for i, child in enumerate(net_chilren):
                    if i == 0:
                        child.register_forward_hook(hook=hook_feature)
                    else:
                        net_chilren_children = child.children()
                        for indirect_children in net_chilren_children:
                            indirect_children.register_forward_hook(hook=hook_feature)
            global  imageindex
            imageindex=0

            for (images, filename, label),(images_n, filename_n, label_n) in zip(data_loader,data_loader_neural):
                print(index)
                imageindex+=1
                label = label.cuda()
                images = images.cuda()
                images_n =images_n.cuda()

                # demo=group_pca_color_augmention(images)
                # save_img(demo, filename, opt.output_dir)

                # Start Attack

                if method=="neural_transfer":
                    adv_img=neural_TDSMFGSM(models[surrogate], images, images_n,label,not 'resnet' in surrogate)
                elif method=="neural_transfer_MIFGSM_test":
                    adv_img=neuraltransfer_MIFGSM(models[surrogate], images, images_n,label,not 'resnet' in surrogate)
                elif method=="neural_transfer_benign":
                    adv_img=images_n
                elif method.startswith('utimate'):

                    adv_img = attack(models[surrogate], images, label, surrogate, method=method,images_n=images_n)
                else:
                    # set_trace()
                    adv_img = attack(models[surrogate], images, label,surrogate,method=method)
                    # set_trace()
                    # save_img(adv_img, filename, opt.output_dir)
                # Save adversarial examples
                # save_img(adv_img, filename, method)

                # Prediction
                if opt.visualize:
                    break
                with torch.no_grad():
                    for net in list_nets:
                        logits[net] = models[net](adv_img)
                        correct_num[net] += (torch.argmax(logits[net][0], axis=1) != label).detach().sum().cpu()

            # Print attack success rate
            if not opt.visualize:
                for net in list_nets:
                    df_res.loc[index,'name']=f'transferability_{surrogate}to{net}_{method}'
                    df_res.loc[index, 'dst_benign_success'] = round((correct_num[net]/input_num).item(),3)
                    index+=1
                    print('{} attack {} using {}success rate: {:.2%}'.format(surrogate,net,method, correct_num[net]/input_num))

            if not opt.visualize:
                # df.to_csv(f"./transferability_result/{surrogate}_{opt.method}_loss.csv")

                df_res.to_csv(f"./{opt.method}.csv")

if __name__ == '__main__':
    seed_torch(0)
    global randomseed
    randomseed = 0
    # set_trace()
    main()