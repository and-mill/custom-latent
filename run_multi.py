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
import pickle


# ---------------------------- ARGS ----------------------------
parser = argparse.ArgumentParser(description='tree-ring_inversion_experiment')
parser.add_argument('--model_id', default='stabilityai/stable-diffusion-2-1-base', type=str)
parser.add_argument('--prompt', default="a tree on a field at night", type=str) #'Painting of cute rat that admires itself in the mirror'
parser.add_argument('--guidance_scale', default=7.5, type=float)  # default setting. Allows for precise inversion while still allowing for enough prompt guidance
parser.add_argument('--num_inference_steps', default=50, type=int)
parser.add_argument('--image_length', default=512, type=int)
parser.add_argument('--num_images', default=1, type=int)
parser.add_argument('--seed', default=1337, type=int)
parser.add_argument('--wandb_experiment_name', default='custom_latent', type=str)  # wandb project name
parser.add_argument('--with_wandb', action='store_false', default=False)  # set --no_wandb to disable. Can also set env var WANDB_MODE=dryrun for same effect

# for saving locally
parser.add_argument('--dir_name', default='out', type=str)
parser.add_argument('--dir_in_name', default='in', type=str)
parser.add_argument('--file_prefix', default='wm', type=str)

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


'''
load a batch as list, produce an image for every one, safe the stuff in some appropriate form.
'''


#

# ---------------------------- GET latents ----------------------------
# TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO TODO <----------------- TODO
def load_zT(shape: tuple,path=args.dir_in_name,number = args.num_images):
    """
    This is the initial noise vector that is used to generate the image
    """
    with open(args.dir_in_name+"/latents.pkl","rb") as f1:
        latents_raw = pickle.load(f1)

    rtn = []
    for m_id,l_list in latents_raw.items():
        for l in l_list:
            zt = torch.from_numpy(l)
            zt = zt.to(torch.float32)
            zt = zt.unsqueeze(0)
            rtn.append((m_id,zt))
    if number > len(rtn):
        return rtn
    else:
        return rtn[:number]
# SD 1+2 have 4 latent channels
# image size 512, 512 is 64, 64 in latents




# ---------------------------- GENERATE Image ----------------------------
def generate(args,pipe,zT):        
    out = pipe(args.prompt, num_images_per_prompt=1, guidance_scale=args.guidance_scale,
           height=args.image_length, width=args.image_length,
           num_inference_steps=args.num_inference_steps,
           latents=zT,  # expects torch of size (batch_dim, num_latents_channels, width, height)
    )
    # [0] because we only have one image if args.num_images == 1
    # Need to rewrite code in many places if args.num_images > 1
    return out.images[0]  # is a PIL Image


# ---------------------------- APPLY Distortion ----------------------------
# img remains unchanged for default args
def distort(args, img):
    img_before_distortion = copy.deepcopy(img)  # is a PIL Image
    return img_before_distortion, utils.image_distortion(img, args)  # is a PIL Image


# ---------------------------- DDIM Inversion ----------------------------

# del(pipe)  # free up memory if necessary
def invert(args,img):
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
    return inv_zT.squeeze(0)


# ---------------------------- SAVE to file ----------------------------
def safe(args,img_index,img,img_before_distortion,inv_zT):
    '''
    image naming: prefix_mid_id
    '''
    os.makedirs(args.dir_name, exist_ok=True)

    # images
    img.save(f"{args.dir_name}/{args.file_prefix}{img_index}_img.jpg")
    img_before_distortion.save(f"{args.dir_name}/{args.file_prefix}{img_index}_img_undistorted.jpg")

    # z noises
    np.save(
        file= f"{args.dir_name}/{args.file_prefix}{img_index}_inv_zT.npy",
        arr=inv_zT.cpu().numpy()
    )
    



# --------------------------------------------------------------------
# ------------------------------- MAIN -------------------------------
# --------------------------------------------------------------------

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

#get the latents from file
latents = load_zT(shape=(1, 4, 64, 64),path=args.dir_in_name, number=args.num_images)  # is a torch tensor

#process all of them
for i,lat in enumerate(latents):
    print(i)
    m_id, l = lat
    l = l.to(device)
    img = generate(args,pipe,l)
    img_undistorted, img = distort(args,img)
    inv_zT = invert(args,img)
    safe(args=args,
         img_index="_mid"+str(m_id)+"_"+str(i),
         img=img,
         img_before_distortion=img_undistorted,
         inv_zT =inv_zT)
        