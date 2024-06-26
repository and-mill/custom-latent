# custom-latent

create wandb account

Fetch an access token from wandb website. Think this is needed later for cli-login

```
conda create -n "custom_latent_0" python=3.9
conda activate custom_latent_0

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt

wandb login

sh run.sh
```
