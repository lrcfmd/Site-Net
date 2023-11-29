from os import sched_yield
import pytorch_lightning as pl
import sys
import json
import sys
import datetime
from ray.tune.suggest import NevergradSearch
import torch
import pymatgen
from matminer.featurizers.site import *
import pickle as pk
import matminer
import numpy as np
import numpy
import h5py
from joblib.externals.loky import set_loky_pickler
from joblib import parallel_backend
from joblib import Parallel, delayed
from joblib import wrap_non_picklable_objects
from torch.nn.utils.rnn import pad_sequence
import yaml
from functools import partial


site_feauturizers_dict = matminer.featurizers.site.__dict__
from tqdm import tqdm
import tqdm
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from lightning_module import Attention_Infomax, DIM_h5_Data_Module
from lightning_module import basic_callbacks

# RayTune
import dill
import shutil
import tempfile
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.cloud_io import load as pl_load
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import (
    TuneReportCallback,
    TuneReportCheckpointCallback,
)
from ray.tune.integration.pytorch_lightning import TuneReportCallback
import itertools as iter
from h5_handler import torch_h5_cached_loader
from multiprocessing import cpu_count
from ray.tune.suggest.hyperopt import HyperOptSearch
from nevergrad.optimization import optimizerlib
from ray.tune.suggest.nevergrad import NevergradSearch
import nevergrad as ng
import ray
from datetime import datetime


def train_model(config, data=None, checkpoint_dir=None):
    tune_report_callback = TuneReportCallback(
        {
            "val_loss": "avg_val_loss",
        },
        on="validation_end",
    )
    tune_checkpoint_callback_val = TuneReportCheckpointCallback(
        metrics={"val_loss": "avg_val_loss"},
        filename="checkpoint",
        on="validation_end",
    )

    trainer = pl.Trainer(
        logger=TensorBoardLogger(save_dir=tune.get_trial_dir(), name="", version="."),
        gpus=1,
        callbacks=[
            basic_callbacks(),
            tune_report_callback,
            tune_checkpoint_callback_val,
        ],
        **config["Trainer kwargs"],
        auto_select_gpus=True,
        gradient_clip_val=0.1,
        precision=16,
        amp_level="O1"
    )
    # config["pre_pool_layers"] = [
    #     config["pre_pool_layers_size"] for _ in range(config["pre_pool_layers_n"])
    # ]
    # config["post_pool_layers"] = [
    #     config["post_pool_layers_size"] for _ in range(config["post_pool_layers_n"])
    # ]
    config["pre_pool_layers"] = [config["pre_pool_layers_size"]] * config[
        "pre_pool_layers_n"
    ]
    config["post_pool_layers"] = [config["post_pool_layers_size"]] * config[
        "post_pool_layers_n"
    ]
    config["site_feature_mean"] = data.site_feature_mean
    config["site_feature_std"] = data.site_feature_std
    if checkpoint_dir:
        ckpt = os.path.join(checkpoint_dir, "checkpoint")
        model = Attention_Infomax.load_from_checkpoint(ckpt)
    else:
        model = Attention_Infomax(config)
    print(config)
    trainer.fit(model, data)


def train_infomax_asha_ng(config, Dataset, scheduler):
    ray.init()
    reporter = CLIReporter(metric_columns=["val_loss", "training_iteration"])
    resources_per_trial = {"cpu": cpu_count(), "gpu": 1}
    ng_kwargs = {
        "embedding_size": ng.p.Scalar(lower=1, upper=512).set_integer_casting(),
        "attention_dim_per_head": ng.p.Scalar(lower=1, upper=64).set_integer_casting(),
        "attention_blocks": ng.p.Scalar(lower=1, upper=12).set_integer_casting(),
        "attention_heads": ng.p.Scalar(lower=1, upper=32).set_integer_casting(),
        "pre_pool_layers_n": ng.p.Scalar(lower=1, upper=8).set_integer_casting(),
        "post_pool_layers_n": ng.p.Scalar(lower=1, upper=8).set_integer_casting(),
        "pre_pool_layers_size": ng.p.Scalar(lower=1, upper=1024).set_integer_casting(),
        "post_pool_layers_size": ng.p.Scalar(lower=1, upper=1024).set_integer_casting(),
        # "Batch_Size": ng.p.Scalar(lower=16, upper=1024).set_integer_casting(),
        "Learning_Rate": ng.p.Log(lower=1e-5, upper=1e-1),
        # "activation_function": ng.p.Choice(choices=["relu", "mish"]),
    }
    config["pre_pool_layers_n"] = len(config["pre_pool_layers"])
    config["pre_pool_layers_size"] = sum(config["pre_pool_layers"]) / len(
        config["pre_pool_layers"]
    )
    config["post_pool_layers_n"] = len(config["post_pool_layers"])
    config["post_pool_layers_size"] = sum(config["post_pool_layers"]) / len(
        config["post_pool_layers"]
    )
    ng_space = ng.p.Dict(**ng_kwargs)
    ng_space.value = {key: config[key] for key in ng_kwargs.keys()}
    search_algorithm = NevergradSearch(
        optimizer=ng.optimizers.OnePlusOne,
        mode="min",
        metric="val_loss",
        space=ng_space,
    )

    analysis = tune.run(
        tune.with_parameters(train_model, data=Dataset),
        resources_per_trial=resources_per_trial,
        progress_reporter=reporter,
        scheduler=scheduler,
        config=config,
        raise_on_failed_trial=False,
        max_failures=0,
        num_samples=100,
        search_alg=search_algorithm,
        name=config["pipeline_name"] + "_" + date_time,
        mode="min",
        metric="val_loss",
        local_dir="./results_test",
    )
    print("Best hyperparameters found were: ", analysis.best_config)
    df = analysis.results_df
    df.to_csv("analysis_tune.csv")


if __name__ == "__main__":
    os.environ["TUNE_DISABLE_STRICT_METRIC_CHECKING"] = "1"
    date_time = datetime.now().strftime("%m-%d-%Y_%H:%M:%S")
    print(date_time)
    try:
        print(sys.argv[1])
        with open(str("config/" + sys.argv[1]) + ".yaml", "r") as config_file:
            config = yaml.load(config_file, Loader=yaml.FullLoader)
    except Exception as e:
        print(e)
        raise RuntimeError(
            "Config not found or unprovided, a configuration JSON path is REQUIRED to run"
        )
    width_list = [
        16,
        32,
        64,
        128,
        256,
        512,
    ]
    length_list = [1, 2, 3, 4, 5, 6, 7, 8]
    layer_lists = [[i] * j for i, j in iter.product(width_list, length_list)]
    # tune_config = {
    # "embedding_size": tune.choice([1, 2, 4, 8, 16, 32, 64, 128, 256]),
    # "attention_dim_per_head": tune.choice([1, 2, 4, 8, 16, 32, 64, 128, 256]),
    # "attention_blocks": tune.choice([1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64]),
    # "num_heads": tune.choice([1, 2, 4, 8, 16, 32, 64]),
    # "pre_pool_layers_n": tune.choice(length_list),
    # "post_pool_layers_n": tune.choice(length_list),
    # "pre_pool_layers_size": tune.choice(width_list),
    # "post_pool_layers_size": tune.choice(width_list),
    # "activation_function": tune.choice(["relu", "mish"]),
    # }
    # config = {**config, **tune_config}
    import pathlib

    config["h5_file"] = str(pathlib.Path().absolute()) + "/" + config["h5_file"]
    Dataset = DIM_h5_Data_Module(
        config,
        max_len=100,
        ignore_errors=True,
        overwrite=False,
    )
    PBT = PopulationBasedTraining(
        time_attr="training_iteration",
        perturbation_interval=5,
        hyperparam_mutations={
            "embedding_size": lambda: None,
            "attention_dim_per_head": lambda: None,
            "attention_blocks": lambda: None,
            "num_heads": lambda: None,
            "pre_pool_layers": lambda: None,
            "post_pool_layers": lambda: None,
        },
    )

    ASHA = ASHAScheduler(grace_period=1, max_t=5)
    train_infomax_asha_ng(config, Dataset, ASHA)
