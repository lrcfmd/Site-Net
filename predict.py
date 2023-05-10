import pytorch_lightning as pl
import sys
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
import os
import torch
import pandas as pd
import numpy as np
import sys, os
from modules import SiteNetAttentionBlock,SiteNetEncoder,k_softmax
from tqdm import tqdm
from lightning_module import collate_fn
from lightning_module import af_dict as lightning_af_dict
from torch_scatter import segment_coo,segment_csr
from torch import nn
import pickle as pk
#monkeypatches
class TReLU(torch.autograd.Function):
    """
    A transparent version of relu that has a linear gradient but sets negative values to zero,
     used as the last step in band gap prediction to provide an alternative to relu which does not kill gradients
      but also prevents the model from being punished for negative band gap predictions as these can readily be interpreted as zero
    """

    @staticmethod
    def forward(ctx, input):
        """
        f(x) is equivalent to relu
        """
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        """
        f'(x) is linear
        """
        return grad_output

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
compression_alg = "gzip"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ml options")
    parser.add_argument("-c", "--config", default=None)
    parser.add_argument("-f", "--h5_file_name", default=None)
    parser.add_argument("-n", "--limit", default=None,type=int)
    parser.add_argument("-m", "--model_name", default=None,type=str)
    parser.add_argument("-w", "--number_of_worker_processes", default=1,type=int)
    parser.add_argument("-d", "--data_csv", default=None,type=str) #Output predictions to custom cif input csv if given
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
    results_list = []
    model_name = args.model_name
    dataset_name = args.h5_file_name
    config["h5_file"] = dataset_name
    config["Max_Samples"] = args.limit
    config["dynamic_batch"] = False
    config["Batch_Size"] = 128
    model = SiteNet(config)
    model.load_state_dict(torch.load(model_name,map_location=torch.device("cpu"))["state_dict"], strict=False)
    Dataset = DIM_h5_Data_Module(
        config,
        max_len=None,
        ignore_errors=True,
        overwrite=False,
        cpus=args.number_of_worker_processes,
        chunk_size=32
    ) 
    print([i["file_key"] for i in Dataset.Dataset])
    results = model.forward(Dataset.Dataset,return_truth=True,batch_size=128)
    predictions = results[0].numpy().flatten()
    truth = results[1].numpy().flatten()
    MAE = np.abs(truth-predictions)
    print("MAE is " + str(MAE.mean()))

    if args.data_csv:
        print("CSV file provided, this CSV file should be the CSV used to generate the hdf5 file from a zip of cifs, the predictions will be added as a third column")
        df = pd.read_csv(args.data_csv,index_col="file")
        df["Prediction"] = pd.Series(predictions,[i["file_key"] for i in Dataset.Dataset]) #Indicies for the prediction pandas series are aligned with the 
        df.to_csv(args.data_csv)
        print("Predictions saved to CSV file!")

    




    
