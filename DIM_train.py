import pytorch_lightning as pl
import sys
from matminer.featurizers.site import *
import matminer
site_feauturizers_dict = matminer.featurizers.site.__dict__
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from lightning_module import (
    basic_callbacks,
    DIM_h5_Data_Module,
    SiteNet,
    SiteNet_DIM
)
from lightning_module import basic_callbacks
import yaml
from h5_handler import torch_h5_cached_loader
from pytorch_lightning.callbacks import *
import argparse
from compress_pickle import dump, load
import collections.abc as container_abcs
from pytorch_lightning.callbacks import ModelCheckpoint

compression_alg = "gzip"

def train_model(config, Dataset):
    if int(args.load_checkpoint) == 1:
        print(config["h5_file"])
        resume_from_checkpoint = args.fold_name + str(config["label"]) + ".ckpt"
    else:
        resume_from_checkpoint = None
    trainer = pl.Trainer(
        gpus=int(args.num_gpus),
        callbacks=[
            basic_callbacks(filename=args.fold_name + str(config["label"])),
        ],
        **config["Trainer kwargs"],
        auto_select_gpus=True,
        detect_anomaly=False,
        #gradient_clip_algorithm="value",
        log_every_n_steps=500,
        val_check_interval=1.0,
        precision=16,
        #amp_level="O2",
        resume_from_checkpoint=resume_from_checkpoint,
    )
    model = SiteNet_DIM(config)
    trainer.fit(model, Dataset)


import pickle as pk

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ml options")
    parser.add_argument("-c", "--config", default="test")
    parser.add_argument("-p", "--pickle", default=0)
    parser.add_argument("-l", "--load_checkpoint", default=0)
    parser.add_argument("-g", "--num_gpus", default=1)
    parser.add_argument("-f", "--fold_name", default="null")
    parser.add_argument("-o", "--overwrite", default=False)
    parser.add_argument("-d", "--debug", default=False)
    parser.add_argument("-u", "--unit_cell_limit",default = 100)
    args = parser.parse_args()
    try:
        print(args.config)
        with open(str("config/" + args.config) + ".yaml", "r") as config_file:
            config = yaml.load(config_file, Loader=yaml.FullLoader)
    except Exception as e:
        print(e)
        raise RuntimeError(
            "Config not found or unprovided, a configuration JSON path is REQUIRED to run"
        )
    config["h5_file"] = args.fold_name + ".hdf5"
    if bool(args.debug) == True:
        config["Max_Samples"] = 1000
    if int(args.pickle) == 1:
        print("Loading Pickle")
        Dataset = load(open("db_pickle.pk", "rb"), compression=compression_alg)
        Dataset.batch_size = config["Batch_Size"]
        print("Pickle Loaded")
        print("--------------")
    else:
        Dataset = DIM_h5_Data_Module(
            config,
            max_len=args.unit_cell_limit,
            ignore_errors=False,
            overwrite=bool(args.overwrite),
        )
        if int(args.pickle) == 2:
            dump(Dataset, open("db_pickle.pk", "wb"), compression=compression_alg)
            print("Pickle Dumped")
        if int(args.pickle) == 3:
            dump(Dataset, open("db_pickle.pk", "wb"), compression=compression_alg)
            print("Pickle Dumped")
            sys.exit()
    train_model(config, Dataset)
