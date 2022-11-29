import pytorch_lightning as pl
from matminer.featurizers.site import *
import matminer
site_feauturizers_dict = matminer.featurizers.site.__dict__
from lightning_module import (
    basic_callbacks,
    DIM_h5_Data_Module,
    SiteNet,
)
from lightning_module import basic_callbacks
import yaml
from pytorch_lightning.callbacks import *
import argparse
from pytorch_lightning.callbacks import ModelCheckpoint
import torch

compression_alg = "gzip"

def train_model(config, Dataset):
    if int(args.load_checkpoint) == 1:
        print(config["h5_file"])
        resume_from_checkpoint = args.h5_file_name + str(config["label"]) + ".ckpt"
    else:
        resume_from_checkpoint = None
    checkpoint_callback = ModelCheckpoint(
    monitor="avg_val_loss",
    dirpath="",
    filename=args.h5_file_name + "_best_" + str(config["label"]),
    save_top_k=1,
    mode="min",
)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        callbacks=[
            basic_callbacks(filename=args.h5_file_name + str(config["label"])),
            checkpoint_callback
        ],
        **config["Trainer kwargs"],
        auto_select_gpus=True,
        detect_anomaly=True,
        gradient_clip_val=1,
        #gradient_clip_algorithm="value",
        log_every_n_steps=10000,
        val_check_interval=1.0,
        precision=16,
        #amp_level="O2",
        resume_from_checkpoint=resume_from_checkpoint,
    )
    model = SiteNet(config)
    trainer.fit(model, Dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ml options")
    parser.add_argument("-c", "--config", default=None)
    parser.add_argument("-l", "--load_checkpoint", default=0)
    parser.add_argument("-f", "--h5_file_name", default=None)
    parser.add_argument("-o", "--overwrite", default=False)
    parser.add_argument("-d", "--debug", default=False)
    parser.add_argument("-u", "--unit_cell_limit",default = 100,type=int)
    parser.add_argument("-w", "--number_of_worker_processes", default=1,type=int)
    parser.add_argument("-e", "--experiment_name", default=None)
    args = parser.parse_args()
    torch.set_num_threads(args.number_of_worker_processes)
    try:
        print("config file is " + args.config)
        with open(str(args.config), "r") as config_file:
            config = yaml.load(config_file, Loader=yaml.FullLoader)
    except Exception as e:
        raise RuntimeError(
            "Config not found or unprovided, a path to a configuration yaml must be provided with -c"
        )
    if args.h5_file_name == None:
        raise RuntimeError(
            "h5 file path is None, h5 file path must be provided through -f"
        )
    if args.experiment_name:
        config["label"] = config["label"] + "_" + str(args.experiment_name)
    config["h5_file"] = args.h5_file_name
    if bool(args.debug) == True:
        config["Max_Samples"] = 1000
    Dataset = DIM_h5_Data_Module(
        config,
        max_len=args.unit_cell_limit,
        ignore_errors=False,
        overwrite=bool(args.overwrite),
        cpus=args.number_of_worker_processes,
        chunk_size=32
    )
    train_model(config, Dataset)
