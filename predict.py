from audioop import mul
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
)
from lightning_module import basic_callbacks
import yaml
from h5_handler import torch_h5_cached_loader
from pytorch_lightning.callbacks import *
import argparse
from compress_pickle import dump, load
import os
import torch
import pandas as pd
import numpy as np
import sys, os
from modules import SiteNetAttentionBlock,SiteNetEncoder,k_softmax,set_seq_af_norm,set_norm_dict,af_dict
from tqdm import tqdm
from lightning_module import collate_fn
from lightning_module import af_dict as lightning_af_dict
from torch_scatter import segment_coo,segment_csr
from torch import nn
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

class SiteNetAttentionBlockwreturns(SiteNetAttentionBlock):
    def __init__(
        self, site_dim, interaction_dim, glob_dim,ije_depth = 2,ije_width = 64, heads=4, af="relu", set_norm="batch",tdot=False,k_softmax=-1,attention_hidden_layers=[256,256]
    ):
        super().__init__(site_dim, interaction_dim, glob_dim,ije_depth,ije_width, heads, af, set_norm,tdot,k_softmax,attention_hidden_layers)
    def forward(self, logger,x, interaction_Features, Attention_Mask,Batch_Mask,cutoff_mask=None, m=None):
        #Construct the Bond Features x_ije
        x_i = x[Batch_Mask["attention_i"],:]
        x_j = x[Batch_Mask["attention_j"],:]
        x_ije = torch.cat([x_i, x_j, interaction_Features], axis=2)
        #Construct the Attention Weights
        multi_headed_attention_weights = self.pre_softmax_linear(self.ije_to_multihead(x_ije)) #g^W
        multi_headed_attention_weights[Attention_Mask] = float("-infinity") #Necessary to avoid interfering with the softmax
        if cutoff_mask is not None:
            multi_headed_attention_weights[cutoff_mask] = float("-infinity")
        multi_headed_attention_weights = k_softmax(multi_headed_attention_weights, 1,self.k_softmax) #K_softmax is unused in the paper, ability to disable message passing beyond the highest N coefficients
        logger.append(multi_headed_attention_weights)
        #Compute the attention weights and perform attention
        x = torch.einsum(
            "ijk,ije->iek",
            multi_headed_attention_weights,
            self.ije_to_attention_features(x_ije) #g^F
        )
        #Compute the new site features and append to the global summary
        x= self.global_linear(torch.reshape(x,[x.shape[0],x.shape[1] * x.shape[2],],))
        m = torch.cat([m, x], dim=1) if m != None else x
        #Compute the new interaction features
        New_interaction_Features = self.ije_to_Interaction_Features(x_ije) #g^I
        return x, New_interaction_Features, m, logger
class encoder_with_interaction_returns(SiteNetEncoder):
    def __init__(
        self,
        embedding_size=100,
        site_feature_size=1,
        attention_blocks=4,
        attention_heads=4,
        site_dim_per_head=64,
        global_embedding_dim_per_block_per_head=16,
        pre_pool_layers=[256, 256],
        post_pool_layers=[256, 256],
        activation_function="relu",
        sym_func="mean",
        set_norm="none",
        lin_norm="none",
        interaction_feature_size=3,
        attention_dim_interaction=64,
        tdot=False,
        attention_hidden_layers=[256,256],
        k_softmax=-1,
        distance_cutoff=-1,
        **kwargs,
    ):
        super().__init__(embedding_size,site_feature_size,
        attention_blocks,attention_heads,site_dim_per_head,global_embedding_dim_per_block_per_head,
        pre_pool_layers,post_pool_layers,activation_function,sym_func,set_norm,lin_norm,interaction_feature_size,
        attention_dim_interaction,tdot,attention_hidden_layers,k_softmax,**kwargs)
        self.distance_cutoff=distance_cutoff
        self.Attention_Blocks = nn.ModuleList(
            SiteNetAttentionBlockwreturns(
                site_dim_per_head,
                attention_dim_interaction,
                global_embedding_dim_per_block_per_head,
                af=activation_function,
                heads=attention_heads,
                set_norm=set_norm,
                tdot=tdot,
                k_softmax=k_softmax,
                attention_hidden_layers = attention_hidden_layers
            )
            for _ in range(attention_blocks)
        )
    def forward(
        self,
        Site_Features,
        Interaction_Features,
        Attention_Mask,
        Batch_Mask,
    ):
        # Interaction Feature Dimensions (i,j,Batch,Embedding)
        # Site Feature Dimensions (i,Batch,Embedding)
        # Attention Mask Dimensions (Batch,i), True for padding, False for data

        #Compute Optional attention mask for cut off mode
        if self.distance_cutoff >= 0:
            cutoff_mask = torch.gt(Interaction_Features[:,:,0],self.distance_cutoff)
        else:
            cutoff_mask = None

        logger = []
        global_summary = None
        interaction_Features = self.interaction_featurization_norm(
            self.af(self.interaction_featurization(Interaction_Features))
        )
        Site_Features = self.site_featurization_norm(
            self.af(self.site_featurization(Site_Features))
        )
        logger.append(Interaction_Features)
        for layer in self.Attention_Blocks:
            Site_Features, interaction_Features, global_summary,logger = layer(logger,
                Site_Features, interaction_Features, Attention_Mask,Batch_Mask, cutoff_mask=cutoff_mask, m=global_summary
            )

        for pre_pool_layer, pre_pool_layer_norm in zip(
            self.pre_pool_layers, self.pre_pool_layers_norm
        ):
            global_summary = pre_pool_layer_norm(
                self.af(pre_pool_layer(global_summary))
            )
        if self.sym_func == "mean":
            Global_Representation = segment_coo(global_summary,Batch_Mask["COO"],reduce="mean")
        elif self.sym_func == "max":
            Global_Representation = segment_coo(global_summary,Batch_Mask["COO"],reduce="max")
        else:
            raise Exception()
        for post_pool_layer, post_pool_layer_norm in zip(
            self.post_pool_layers, self.post_pool_layers_norm
        ):
            Global_Representation = post_pool_layer_norm(
                self.af(post_pool_layer(Global_Representation))
            )

        return Global_Representation,logger
class lightning_module_with_interaction_returns(SiteNet):
    def __init__(
        self,
        config=None,
    ):
        super().__init__(config)
        self.encoder = encoder_with_interaction_returns(**self.config)
    def forward(self, b, batch_size=16,return_truth = False):
        self.eval()
        lob = [b[i : min(i + batch_size,len(b))] for i in range(0, len(b), batch_size)]
        Encoding_list = []
        targets_list= []
        Logger_list = []
        print("Inference in batches of %s" % batch_size)
        for inference_batch in tqdm(lob):
            batch_dictionary = collate_fn(inference_batch, inference=True)
            Attention_Mask = batch_dictionary["Attention_Mask"]
            Site_Feature = batch_dictionary["Site_Feature_Tensor"]
            Atomic_ID = batch_dictionary["Atomic_ID"]
            interaction_Features = batch_dictionary["Interaction_Feature_Tensor"]
            Oxidation_State = batch_dictionary["Oxidation_State"]
            Batch_Mask = batch_dictionary["Batch_Mask"]
            concat_embedding = self.input_handler(
                Atomic_ID, [Site_Feature, Oxidation_State]
            )
            with torch.no_grad():
                Encoding,logger = self.encoder.forward(
                    concat_embedding,
                    interaction_Features,
                    Attention_Mask,
                    Batch_Mask,
                )
                logger.append(batch_dictionary["Structure"])
                Encoding = lightning_af_dict[self.config["last_af_func"]](self.decoder(Encoding))
                if self.config["regularization strategy"] == "l1_sparse":
                    Encoding = F.relu(Encoding)
                if self.config["regularization strategy"] == "kl_sparse":
                    Encoding = F.relu(Encoding)
                Encoding_list.append(Encoding)
                targets_list.append(batch_dictionary["target"])
                Logger_list.append(logger)
        Encoding = torch.cat(Encoding_list, dim=0)
        targets = torch.cat(targets_list, dim=0)
        self.train()
        if return_truth:
            return [Encoding,targets,Logger_list]
        else:
            return Encoding
    

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


def wrapped(pers_data):
    try:
        print(sys.argv[1])
        with open(str("config/" + sys.argv[1]) + ".yaml", "r") as config_file:
            config = yaml.load(config_file, Loader=yaml.FullLoader)
    except Exception as e:
        print(e)
        raise RuntimeError(
            "Config not found or unprovided, a configuration JSON path is REQUIRED to run"
        )
    Dataset = DIM_h5_Data_Module(
        config, max_len=100, ignore_errors=True, overwrite=False, Dataset=pers_data
    )
    train_model(config, Dataset)


def train_model(config, Dataset):
    # torch.autograd.set_detect_anomaly(True)
    if int(args.load_checkpoint) == 1:
        resume_from_checkpoint = config["h5_file"] + ".ckpt"
    else:
        resume_from_checkpoint = None
    trainer = pl.Trainer(
        gpus=int(args.num_gpus),
        callbacks=[
            basic_callbacks(filename=config["h5_file"]),
            LearningRateMonitor(logging_interval="step"),
        ],
        **config["Trainer kwargs"],
        auto_select_gpus=True,
        gradient_clip_val=1,
        log_every_n_steps=1,
        val_check_interval=0.25,
        precision=16,
        amp_level="O2",
        resume_from_checkpoint=resume_from_checkpoint,
    )
    model = SiteNet(config)
    trainer.fit(model, Dataset)


import pickle as pk

if __name__ == "__main__":
    torch.set_num_threads(32)
    parser = argparse.ArgumentParser(description="ml options")
    parser.add_argument("-c", "--config", default="test")
    parser.add_argument("-p", "--pickle", default=0)
    parser.add_argument("-l", "--load_checkpoint", default=0)
    parser.add_argument("-g", "--num_gpus", default=1)
    parser.add_argument("-d", "--dataset", default="null")
    parser.add_argument("-n", "--limit", default=None,type=int)
    parser.add_argument("-m", "--model_name", default=None,type=str)
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
    models = [i for i in os.listdir("Data/CV_hdf5") if args.dataset in i  and ".ckpt" in i]
    results_list = []
    for fold in range(0,1):
        model_name = args.model_name
        print(model_name)
        dataset_name = args.dataset + "_test_" + str(fold+1) + ".hdf5"
        config["h5_file"] = "Data/CV_hdf5/" + dataset_name
        config["Max_Samples"] = args.limit
        config["dynamic_batch"] = False
        config["Batch_Size"] = 128
        print(config["h5_file"])
        print(args.limit)
        model = lightning_module_with_interaction_returns(config)
        model.load_state_dict(torch.load("Data/CV_hdf5/" + model_name,map_location=torch.device("cpu"))["state_dict"])
        #for name, param in model.named_parameters():
            #print(name)
            #print(param)
        Dataset = DIM_h5_Data_Module(
            config,
            max_len=None,
            ignore_errors=True,
            overwrite=False,
        )  
        results = model.forward(Dataset.Dataset,return_truth=True,batch_size=128)

        predictions = results[0].numpy().flatten()
        truth = results[1].numpy().flatten()
        MAE = np.abs(truth-predictions)
        print(MAE.mean())
        results_df = pd.DataFrame([predictions,truth,MAE]).transpose()
        results_df.to_csv("parity plot data test " + str(fold + 1) + ".csv")
        results_list.append(results_df)
        torch.save(model.state_dict(),"Data/CV_hdf5/e_gap_train_1_test.sdic")
        MAE_clamped = np.abs(truth-np.clip(predictions,0,None))     
        len_1 = len(results[2])
        len_2 = len(results[2][0])
        attention_logs = [[results[2][j][i] for j in range(len_1)] for i in range(len_2-1)]
        attention_logs = [[j.flatten(end_dim=1) for j in i] for i in attention_logs]
        attention_logs = [torch.cat(i,dim=0) for i in attention_logs]
        attention_logs = torch.cat(attention_logs,dim=1)
        t_np = attention_logs.numpy() #convert to Numpy array
        df = pd.DataFrame(t_np,columns=["x1","x2","y11","y12","y13","y21","y22","y23"]) #convert to a dataframe
        df.to_csv("attention_logs.csv",index=False) #save to file

    
    
    
    
