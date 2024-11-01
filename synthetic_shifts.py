import albumentations as alb
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
from torchvision.transforms import ToTensor
from torch.utils import data
import albumentations
import random


def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
def additive_noise(x, intensity):
    seed_all(0)
    noise = torch.randn_like(x) * intensity
    return x + noise

def desaturate(img, intensity):
    seed_all(0)
    intensity_normed = np.clip(intensity*0.5/0.3, -0.5, 0.5)
    x = img.permute(1, 2, 0).numpy()
    desat = albumentations.ColorJitter(hue=0, brightness=0, saturation=[intensity_normed, intensity_normed], contrast=0, always_apply=True)
    transforms = alb.Compose([desat])
    transformed = transforms(image=x)["image"]
    transformed = ToTensor()(transformed)
    return transformed
def hue_shift(img, intensity):
    seed_all(0)
    x = img.permute(1, 2, 0).numpy()

    intensity_normed = np.clip(intensity*0.5/0.3, -0.5, 0.5)

    desat = albumentations.ColorJitter(hue=[intensity_normed, intensity_normed], brightness=0, saturation=0, contrast=0, always_apply=True)
    transforms = alb.Compose([desat])
    transformed = transforms(image=x)["image"]
    transformed = ToTensor()(transformed)
    return transformed

def brightness_shift(img, intensity):
    seed_all(0)
    x = img.permute(1, 2, 0).numpy()
    intensity_normed = np.clip(intensity*0.5/0.3, -0.5, 0.5)

    desat = albumentations.ColorJitter(brightness=[intensity_normed, intensity_normed], hue=0, contrast=0, saturation=0, always_apply=True)
    transforms = alb.Compose([desat])
    transformed = transforms(image=x)["image"]
    transformed = ToTensor()(transformed)
    return transformed

def contrast_shift(img, intensity):
    seed_all(0)
    x = img.permute(1, 2, 0).numpy()
    intensity_normed = np.clip(intensity*0.5/0.3, -0.5, 0.5)

    desat = albumentations.ColorJitter(contrast=[intensity_normed, intensity_normed], hue=0, brightness=0, saturation=0, always_apply=True)
    transforms = alb.Compose([desat])
    transformed = transforms(image=x)["image"]
    transformed = ToTensor()(transformed)
    return transformed

def multiplicative_noise(x, intensity):
    seed_all(0)
    noise = 1+torch.randn_like(x) * intensity*2
    return x * noise

def salt_and_pepper(x, intensity):
    seed_all(0)
    noise = torch.rand_like(x)
    x = x.clone()
    x[noise<intensity] = 0
    x[noise>1-intensity] = 1
    return x

def smear(batch, intensity):
    seed_all(0)
    x = batch[0].permute(1, 2, 0).numpy()
    intensity_normed = np.clip(intensity*0.5/0.3, -0.5, 0.5)
    desat = albumentations.GridDistortion(always_apply=True, distort_limit=[intensity_normed, intensity_normed])
    transforms = alb.Compose([desat])

    if len(batch[1].shape)==3:
        # print("seg")
        y = batch[1].permute(1, 2, 0).numpy()

        transformed = transforms(image=x, mask=y)
        new_mask =  ToTensor()(transformed["mask"])
        return ToTensor()(transformed["image"]), new_mask
    else:

        transformed = transforms(image=x, )["image"]
        transformed = ToTensor()(transformed)
        return transformed, *batch[1:]


def fgsm(img, intensity, model):
    seed_all(0)
    x = img.cuda().unsqueeze(0).requires_grad_(True)
    model.eval()
    output = model(x)
    if len(output.shape)==2: #classification
        loss = model.criterion(output, output.argmax(dim=1)).mean()
        model.zero_grad()
        loss.backward()
        data_grad = x.grad.data
        perturbed_x = x + intensity * data_grad.sign()
        perturbed_x = torch.clamp(perturbed_x, 0, 1)
        return perturbed_x.squeeze()

    else:
        loss = model.criterion(output, torch.ones_like(output).cuda()).mean()
        model.zero_grad()
        loss.backward()
        data_grad = x.grad.data
        perturbed_x = x - intensity * data_grad.sign()
        perturbed_x = torch.clamp(perturbed_x, 0, 1)
        return perturbed_x.squeeze()


def random_occlusion(img, intensity):
    seed_all(0)
    x = img.permute(1, 2, 0).numpy()
    occlusion = albumentations.Cutout(int(intensity*100), max_h_size=int(x.shape[0]*0.1), max_w_size=int(x.shape[1]*0.1), always_apply=True)
    transforms = alb.Compose([occlusion])
    transformed = transforms(image=x)["image"]
    transformed = ToTensor()(transformed)
    return transformed


class TransformedDataset(data.Dataset):
    #generic wrapper for adding noise to datasets
    def __init__(self, dataset, transform, transform_name, transform_param, model="None"):
        super().__init__()
        self.dataset = dataset
        self.transform = transform
        self.transform_param = transform_param
        self.transform_name = transform_name
        print(transform_name)
        print(transform_param)
        self.model=model
    def __getitem__(self, index):

        batch = self.dataset.__getitem__(index)
        x = batch[0]
        y = batch[1]
        rest = batch[1:]
        if self.transform_name=="fgsm":
            x = torch.clip(self.transform(x, self.transform_param, self.model), 0, 1)
        if self.transform_name =="smear":
            x, y = self.transform(batch, self.transform_param)
            return x, y, batch[2:]
        else:
            x = torch.clip(self.transform(x, self.transform_param), 0, 1)
        if index==0:
            plt.imshow(x.permute(1,2,0).detach().cpu().numpy())
            plt.savefig(f"test_plots/{self.transform_name}_{self.transform_param}.png")
            plt.show()
            plt.close()
        return (x, *rest)

    def __str__(self):
        return f"{type(self.dataset).__name__}_{self.transform_name}_{str(self.transform_param)}"

    def __len__(self):
        # return 1000 #debug
        return self.dataset.__len__()

