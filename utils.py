import torch
from torchvision import transforms

from PIL import Image, ImageFilter
import random
import numpy as np

from diffusers import AutoencoderKL


def set_random_seed(seed: int = 0):
    torch.manual_seed(seed + 0)
    torch.cuda.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 2)
    np.random.seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 4)
    random.seed(seed + 5)


def image_distortion(img: Image, args):
    """
    Distort one PIL image
    See args in run.py for the list of possible distortions
    """
    if args.r_degree is not None:
        img = transforms.RandomRotation((args.r_degree, args.r_degree))(img)

    if args.jpeg_ratio is not None:
        img.save(f"tmp1_{args.jpeg_ratio}_{args.run_name}.jpg", quality=args.jpeg_ratio)
        img = Image.open(f"tmp1_{args.jpeg_ratio}_{args.run_name}.jpg")

    if args.crop_scale is not None and args.crop_ratio is not None:
        set_random_seed(args.seed)
        img = transforms.RandomResizedCrop(img.size, scale=(args.crop_scale, args.crop_scale), ratio=(args.crop_ratio, args.crop_ratio))(img)
        
    if args.gaussian_blur_r is not None:
        img = img.filter(ImageFilter.GaussianBlur(radius=args.gaussian_blur_r))

    if args.gaussian_std is not None:
        img_shape = np.array(img).shape
        g_noise = np.random.normal(0, args.gaussian_std, img_shape) * 255
        g_noise = g_noise.astype(np.uint8)
        img = Image.fromarray(np.clip(np.array(img) + g_noise, 0, 255))

    if args.brightness_factor is not None:
        img = transforms.ColorJitter(brightness=args.brightness_factor)(img)

    return img



def pil_2_tensor(pil: Image):
    """
    Convert single PIL image to tensor
    """
    return transforms.ToTensor()(pil).unsqueeze(0)


def tensor_2_pil(tensor: torch.Tensor):
    """
    Convert single tensor to PIL image
    """
    return transforms.ToPILImage()(tensor.squeeze(0))


def img_to_latents(x: torch.Tensor, vae: AutoencoderKL):
    """
    Get latents from image given some autoencoder
    """
    x = 2. * x - 1.
    posterior = vae.encode(x).latent_dist
    latents = posterior.mean * 0.18215
    return latents
