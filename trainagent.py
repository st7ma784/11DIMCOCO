


   
   
import wandb
from trainclip_v2 import train
import argparse
import wandb
from functools import partial
wandb.login()
parser = argparse.ArgumentParser()
parser.add_argument("--sweep_id", default="futbumne",nargs="?", type=str)
parser.add_argument("--data_dir", default="/Data",nargs="?", type=str)
parser.add_argument("--devices", default="auto",nargs="?", type=str)
parser.add_argument("--accelerator", default="auto",nargs="?", type=str)

p = parser.parse_args()
train=partial(train,dir=p.data_dir,devices=p.devices, accelerator=p.accelerator)
wandb.agent(sweep_id=p.sweep_id, project="6DimContrSweep", entity="st7ma784",function=train)
