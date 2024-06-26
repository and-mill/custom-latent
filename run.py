import os

import argparse
import wandb
import copy

import torch
import numpy as np

import PIL
from PIL import Image

from diffusers import DDIMScheduler, DDIMInverseScheduler, StableDiffusionPipeline

import utils



# ---------------------------- ARGS ----------------------------
parser = argparse.ArgumentParser(description='tree-ring_inversion_experiment')
parser.add_argument('--model_id', default='stabilityai/stable-diffusion-2-1-base', type=str)
parser.add_argument('--prompt', default='Painting of cute rat that admires itself in the mirror', type=str)
parser.add_argument('--guidance_scale', default=7.5, type=float)  # default setting. Allows for precise inversion while still allowing for enough prompt guidance
parser.add_argument('--num_inference_steps', default=50, type=int)
parser.add_argument('--image_length', default=512, type=int)
parser.add_argument('--num_images', default=1, type=int)
parser.add_argument('--seed', default=1337, type=int)
parser.add_argument('--wandb_experiment_name', default='custom_latent', type=str)  # wandb project name
parser.add_argument('--with_wandb', action='store_false', default=True)  # set --no_wandb to disable. Can also set env var WANDB_MODE=dryrun for same effect

# for saving locally
parser.add_argument('--dir_name', default='out', type=str)
parser.add_argument('--file_prefix', default='run0_', type=str)

# for image distortion
parser.add_argument('--r_degree', default=0, type=float)
parser.add_argument('--jpeg_ratio', default=None, type=int)
parser.add_argument('--crop_scale', default=None, type=float)
parser.add_argument('--crop_ratio', default=None, type=float)
parser.add_argument('--gaussian_blur_r', default=None, type=int)
parser.add_argument('--gaussian_std', default=None, type=float)
parser.add_argument('--brightness_factor', default=None, type=float)
parser.add_argument('--rand_aug', default=0, type=int)

args, unknown = parser.parse_known_args()

utils.set_random_seed(args.seed)


# ---------------------------- LOAD Model ----------------------------
# load diffusion model
if not torch.cuda.is_available():
    raise Exception('CUDA not available')
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# original pipe (used for generating images and also inversion)
pipe = StableDiffusionPipeline.from_pretrained(
    args.model_id,
    scheduler=DDIMScheduler.from_pretrained(args.model_id, subfolder='scheduler'),
    torch_dtype=torch.float32,
    variant='fp16',
    safety_checker=None, requires_safety_checker=False,
    ).to(device)


# ---------------------------- GET latents ----------------------------
# TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO <----------------- TODO
def get_zT(shape: tuple):
    """
    This is the initial noise vector that is used to generate the image
    """
    return torch.randn(shape)

# SD 1+2 have 4 latent channels
# image size 512, 512 is 64, 64 in latents
zT = get_zT(shape=(args.num_images, 4, 64, 64)).to(device)  # is a torch tensor


# ---------------------------- GENERATE Image ----------------------------
        
out = pipe(args.prompt, num_images_per_prompt=args.num_images, guidance_scale=args.guidance_scale,
           height=args.image_length, width=args.image_length,
           num_inference_steps=args.num_inference_steps,
           latents=zT,  # expects torch of size (batch_dim, num_latents_channels, width, height)
    )
# [0] because we only have one image if args.num_images == 1
# Need to rewrite code in many places if args.num_images > 1
img = out.images[0]  # is a PIL Image


# ---------------------------- APPLY Distortion ----------------------------
# img remains unchanged for default args
img_before_distortion = copy.deepcopy(img)  # is a PIL Image
img = utils.image_distortion(img, args)  # is a PIL Image


# ---------------------------- DDIM Inversion ----------------------------

# del(pipe)  # free up memory if necessary

inv_pipe = StableDiffusionPipeline.from_pretrained(args.model_id,
                                                   scheduler=DDIMInverseScheduler.from_pretrained(args.model_id, subfolder='scheduler'),
                                                   torch_dtype=torch.float32,
                                                   variant='fp16',
                                                   safety_checker=None, requires_safety_checker=False
                                                   ).to(device)
vae = inv_pipe.vae
z0 = utils.img_to_latents(utils.pil_2_tensor(img).to(device), vae)
inv_zT, _ = inv_pipe(prompt="", negative_prompt="", guidance_scale=1.,
                     width=args.image_length, height=args.image_length,
                     num_inference_steps=args.num_inference_steps,
                     latents=z0,
                     output_type='latent', return_dict=False,)  # is a torch tensor


 # ---------------------------- REMOVE batch dims before saving ----------------------------
zT = zT.squeeze(0)
inv_zT = inv_zT.squeeze(0)


# ---------------------------- SAVE to file ----------------------------
os.makedirs(args.dir_name, exist_ok=True)

# images
img.save(f"{args.dir_name}/{args.file_prefix}_img.jpg")
img_before_distortion.save(f"{args.dir_name}/{args.file_prefix}_img_before_distortion.jpg")

# z noises
torch.save(zT, f"{args.dir_name}/{args.file_prefix}_zT.pt")
#zT = torch.load(f"{args.dir_name}/{args.file_prefix}_zT.pt")
torch.save(inv_zT, f"{args.dir_name}/{args.file_prefix}_inv_zT.pt")
#inv_zT = torch.load(f"{args.dir_name}/{args.file_prefix}_inv_zT.pt")

# ---------------------------- SAVE to Wandb ----------------------------
if args.with_wandb:

    def wandb_type(cell):
        """Helper funktion to convert data to wandb compatible format"""
        if isinstance(cell, torch.Tensor):
            cell = cell.detach().cpu().numpy()
            # Latents often have 4 channels. Save RGBA then
            if cell.shape[0] == 4:
                # Normalize the tensor values to the range [0, 1]
                cell = (cell - cell.min()) / (cell.max() - cell.min())
                cell = (cell * 255).astype(np.uint8)
                # Relocate Channel dim, Convert to PIL image
                cell = np.transpose(cell, (1, 2, 0))
                pil_image = Image.fromarray(cell, 'RGBA')
                # return
                return wandb.Image(pil_image, mode="RGBA")
            else:
                return wandb.Image(cell)
        elif isinstance(cell, PIL.Image.Image):
             return wandb.Image(cell)
        elif isinstance(cell, str):
            return cell
        else:
            raise Exception(f'Unknown type: {type(cell)}')

    # define the output row for this experiment
    row = {'prompt': args.prompt,
           'zT_RGBA': zT,
           'zT_ch0': zT[0],
           'zT_ch1': zT[1],
           'zT_ch2': zT[2],
           'zT_ch3': zT[3],
           'img_before_distortion': img_before_distortion,
           'img': img,
           'inv_zT_RGBA': inv_zT,
           'inv_zT_ch0': inv_zT[0],
           'inv_zT_ch1': inv_zT[1],
           'inv_zT_ch2': inv_zT[2],
           'inv_zT_ch3': inv_zT[3],
           }

    wandb.init(project=args.wandb_experiment_name)
    wandb.config.update(args)
    
    # write table
    columns = list(row.keys())
    table = wandb.Table(columns=columns)
    table.add_data(*[wandb_type(row[col]) for col in row.keys()])  # add data to table and convert each cell to appropriate type
    wandb.log({'table': table})

    wandb.finish()  # this important
