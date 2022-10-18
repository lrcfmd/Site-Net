from ast import Global
import torch
import pytorch_lightning as pl
from h5_handler import *
from torch.utils.data import DataLoader, random_split
from multiprocessing import cpu_count
from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad
from modules import (
    SiteNetEncoder,SiteNetDIMAttentionBlock,SiteNetDIMGlobal
)
import sys
import torch.nn.functional as F
from torch.nn import Softmax as smax
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import *
from torch import nn
from random import randint
from torch_scatter import scatter_std,scatter_mean,segment_coo,segment_csr
from torch.utils.data import BatchSampler,RandomSampler,Sampler,SequentialSampler,SubsetRandomSampler
import random

optim_dict = torch.optim.__dict__
site_feauturizers_dict = matminer.featurizers.site.__dict__

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

#Dictionary of activation functions
af_dict = {"identity":lambda x:x,"relu":nn.functional.relu,"softplus":nn.functional.softplus,"TReLU":TReLU.apply}

class interaction_featurizer:
    def __init__(self, base_function, log=False, polynomial_degree=1, max_clip=25):
        self.base_function = base_function
        self.polynomial = polynomial_degree
        self.log = log
        self.max_clip = max_clip

    def featurize(self, structure):
        base_matrix = self.base_function(structure)
        base_matrix = base_matrix ** self.polynomial
        if self.log:
            base_matrix = np.log(base_matrix)
        base_matrix = np.clip(base_matrix, -self.max_clip, self.max_clip)
        return base_matrix


def distance_matrix(structure, func=lambda _: _):
    distance_matrix = func(structure.distance_matrix)
    return distance_matrix


def coulomb_matrix(structure):
    return structure_feat.CoulombMatrix(diag_elems=False, flatten=False).featurize(
        structure
    )[0]


interaction_featurizer_Functions = {
    "distance_matrix": distance_matrix,
    "reciprocal_square_distance_matrix": lambda structure_l: distance_matrix(
        structure_l, func=lambda _: _ ** -2
    ),
    "coulomb_matrix": coulomb_matrix,
}


def structure_processing_site_features(structure, Site_Featurizers, interaction_featurizers):
    try:
        return_dict = {}
        if Site_Featurizers != None:

            return_dict["Site_Feature_Tensor"] = np.array(
                [
                    np.concatenate(
                        [
                            Featurizer.featurize(structure, i)
                            for Featurizer in Site_Featurizers
                        ],
                        axis=0,
                    )
                    for i in range(len(structure))
                ]
            )
        else:
            return_dict["Site_Feature_Tensor"] = np.empty((len(structure), 0))
        if interaction_featurizers != None:
            return_dict["Interaction_Feature_Tensor"] = np.stack(
                [Featurizer.featurize(structure) for Featurizer in interaction_featurizers],
                axis=2,
            )
        else:
            return_dict["Interaction_Feature_Tensor"] = np.empty(
                (len(structure), len(structure), 0)
            )
        return return_dict

    except Exception as e:
        traceback.print_exc()
        print(e)
        return None

#Constructs a batch dictionary from the list of outputs from the h5 file
def collate_fn(batch, inference=False):
    batch_dict = {}
    # Unpack Crystal Features Dictionaries
    primitive_lengths = [i["prim_size"] for i in batch]
    image_count = [i["images"] for i in batch]
    actual_length = [i["prim_size"]*i["images"] for i in batch]
    #primitive_lengths = [i["prim_size"]*i["images"] for i in batch]
    #image_count = [1 for i in batch]
    #actual_length = [i["prim_size"]*i["images"] for i in batch]
    #[list(range(0,j,k))]
    def batch_permute(batch_full, key, dtype, process_func=lambda _: _):
        batch = [i[key] for i in batch_full]
        batch = [process_func(i) for i in batch]
        batch = [torch.as_tensor(np.array(i), dtype=dtype) for i in batch]
        return batch

    def batch_permute_sparse(batch_full, key, dtype, process_func=lambda _: _):
        batch = [i[key] for i in batch_full]
        batch = [process_func(i) for i in batch]
        batch = [torch.as_tensor(i, dtype=dtype) for i in batch]
        return batch

    def two_d_batch_permute(batch_full, key, dtype, process_func=lambda _: _):
        batch = [i[key] for i in batch_full]
        batch = [process_func(i) for i in batch]
        batch = [torch.as_tensor(i, dtype=dtype) for i in batch]
        return batch

    # 2 dimensional stacking for the adjaceny matricies
    def adjacency_stack(batch):
        max_matrix_dim = max(
            i.shape[1] for i in batch
        )  # Should be of dimensions (i,j,E)
        stacked = torch.cat(
            [
                pad(
                    input=i,
                    pad=(
                        0,
                        0,
                        0,
                        max_matrix_dim - i.shape[1],
                        0,
                        0,
                    ),
                )
                for i in batch
            ],
            0
        )
        return stacked
    Atomic_ID = batch_permute(batch, "Atomic_ID", torch.long)
    site_features = batch_permute_sparse(batch, "Site_Feature_Tensor", torch.float)
    interaction_features = two_d_batch_permute(batch, "Interaction_Feature_Tensor", torch.float)
    Oxidation_State = batch_permute(batch, "Oxidation_State", torch.float)
    # Pack Crystal Features
    batching_mask_COO = []
    batching_mask_CSR = []
    index_CSR = 0
    for idx,i in enumerate(site_features):
        batching_mask_CSR.append(index_CSR)
        index_CSR += i.shape[0]
        batching_mask_COO.extend([idx]*i.shape[0])
    batching_mask_CSR.append(index_CSR)
    array_i = []
    array_j = []
    for idx,batch_n in enumerate(np.unique(batching_mask_COO)):
        item_index = np.where(batching_mask_COO == batch_n)[0]
        prim_count = primitive_lengths[idx]
        images = image_count[idx]
        #Creates the site indicies for the i'th site in the bond feautres for each j
        array_i.append(np.array([[i for _ in range(max(actual_length))] for i in item_index]))
        #Creates the site indicies for the j'th component of the bond feautues for each i
        array_j.append(np.array([[item_index[(i//images)%prim_count] for i in range(max(actual_length))] for _ in item_index]))
    #These masks allows the self attention mechanism to index the correct site features when constructing the bond features, without the use of a batch dimension
    batching_mask_attention_i = torch.tensor(np.concatenate(array_i,0),dtype=torch.long)
    batching_mask_attention_j = torch.tensor(np.concatenate(array_j,0),dtype=torch.long)
    #Different "scattered" functions enabling batching along i require different expressions of the batch indicies
    #COO labels every position along i according to its batch
    #CSR provides the indicies where each new batch begins
    #iterate adds the length of the site features onto the end of the array to enable sliding window iteration
    batching_mask_COO = torch.tensor(batching_mask_COO,dtype=torch.long)
    batching_mask_CSR = torch.tensor(batching_mask_CSR,dtype=torch.long)

    #The attention mask excludes the zero padding along j during attention
    #Actual length is the number of real atomic sites each supercell actually has, the primitive point group multiplied by the number of images we consider
    #Anything past the actual length is junk data and needs to be marked as such
    base_attention_mask = torch.stack(
        [
            torch.tensor(
                [False if i < len else True for i in range(max(actual_length))],
                dtype=torch.bool,
            )
            for len in actual_length
        ]
    )
    batch_dict["Attention_Mask"] = base_attention_mask[batching_mask_COO,:]
    batch_dict["Site_Feature_Tensor"] = torch.cat(site_features,0)
    batch_dict["Interaction_Feature_Tensor"] = adjacency_stack(interaction_features)
    batch_dict["Atomic_ID"] = torch.cat(Atomic_ID,0)
    batch_dict["Oxidation_State"] = torch.cat(Oxidation_State,0)
    batch_dict["target"] = torch.as_tensor(np.array([(i["target"]) for i in batch]))
    batch_dict["Batch_Mask"] = {"COO":batching_mask_COO,"CSR":batching_mask_CSR,"attention_i":batching_mask_attention_i,"attention_j":batching_mask_attention_j}
    if inference:
        batch_dict["Structure"] = [i["structure"] for i in batch]
    return batch_dict

class SiteNet(pl.LightningModule):
    def __init__(
        self,
        config=None,
    ):
        super().__init__()
        if config != None: #Implies laoding from a checkpoint if None
            #Load in the hyper parameters as lightning model attributes
            self.config = config
            self.batch_size = self.config["Batch_Size"]
            self.decoder = nn.Linear(self.config["post_pool_layers"][-1], 1)
            self.encoder = SiteNetEncoder(**self.config)

            self.config["pre_pool_layers_n"] = len(config["pre_pool_layers"])
            self.config["pre_pool_layers_size"] = sum(config["pre_pool_layers"]) / len(
                config["pre_pool_layers"]
            )
            self.config["post_pool_layers_n"] = len(config["post_pool_layers"])
            self.config["post_pool_layers_size"] = sum(config["post_pool_layers"]) / len(
                config["post_pool_layers"]
            )
            #Initialize the learnt elemental embeddings
            self.Elemental_Embeddings = nn.Embedding(
                200,
                self.config["embedding_size"],
                max_norm=1,
                scale_grad_by_freq=False,
            )
            self.save_hyperparameters(self.config)
    #Constructs the site features from the individual pieces, including the learnt atomic embeddings if enabled
    def input_handler(self, atomic_number, features, Learnt_Atomic_Embedding=True):
        if Learnt_Atomic_Embedding:
            Atomic_Embedding = self.Elemental_Embeddings(atomic_number)
        else:
            Atomic_Embedding = F.one_hot(atomic_number, num_classes=115).float()
        for i in features:
            assert not torch.isnan(i).any()
        if torch.isnan(Atomic_Embedding).any():
            print(atomic_number)
            print(Atomic_Embedding)
            raise (Exception)
        return torch.cat([Atomic_Embedding, *features], dim=1)

    #Inference mode, return the prediction
    def forward(self, b, batch_size=16,return_truth = False):
        self.eval()
        lob = [b[i : min(i + batch_size,len(b))] for i in range(0, len(b), batch_size)]
        Encoding_list = []
        targets_list= []
        print("Inference in batches of %s" % batch_size)
        for inference_batch in tqdm(lob):
            batch_dictionary = collate_fn(inference_batch, inference=True)
            Attention_Mask = batch_dictionary["Attention_Mask"]
            Site_Feature = batch_dictionary["Site_Feature_Tensor"]
            Atomic_ID = batch_dictionary["Atomic_ID"]
            Interaction_Features = batch_dictionary["Interaction_Feature_Tensor"]
            Oxidation_State = batch_dictionary["Oxidation_State"]
            Batch_Mask = batch_dictionary["Batch_Mask"]
            concat_embedding = self.input_handler(
                Atomic_ID, [Site_Feature, Oxidation_State]
            )
            with torch.no_grad():
                Encoding = self.encoder.forward(
                    concat_embedding,
                    Interaction_Features,
                    Attention_Mask,
                    Batch_Mask,
                    return_std=False,
                )
                Encoding = af_dict[self.config["last_af_func"]](self.decoder(Encoding))
                if self.config["regularization strategy"] == "l1_sparse":
                    Encoding = F.relu(Encoding)
                if self.config["regularization strategy"] == "kl_sparse":
                    Encoding = F.relu(Encoding)
                Encoding_list.append(Encoding)
                targets_list.append(batch_dictionary["target"])
        Encoding = torch.cat(Encoding_list, dim=0)
        targets = torch.cat(targets_list, dim=0)
        self.train()
        if return_truth:
            return [Encoding,targets]
        else:
            return Encoding

    #Feed in a dataframe with cif files in the "structure" column and get predictions
    def prepare_dataframe(
        self, df, structure_column_name="structure", input_dict_column_name="DIM_input"
    ):
        structures = df[structure_column_name]
        if self.config["Site_Features"] != None:
            Site_Featurizers = [
                site_feauturizers_dict[feat_dict["name"]](
                    *feat_dict["Featurizer_PArgs"], **feat_dict["Featurizer_KArgs"]
                )
                for feat_dict in self.config["Site_Features"]
            ]
        else:
            Site_Featurizers = None
        interaction_featurizers = [
            interaction_featurizer(
                interaction_featurizer_Functions[feat_dict["name"]], **feat_dict["kwargs"]
            )
            for feat_dict in self.config["Interaction_Features"]
        ]
        with multiprocessing.Pool(cpu_count() * 2) as pool:
            iterable = [[i, Site_Featurizers, interaction_featurizers] for i in structures]
            print("calculating site features")
            results = list(
                tqdm(
                    pool.istarmap(
                        structure_processing_site_features,
                        iterable,
                    ),
                    total=len(iterable),
                )
            )
            print("Reading matricies / elemental data from structures")
            results = [
                structure_processing_static(i, j)
                for i, j in tqdm(zip(structures, results))
            ]

        df[input_dict_column_name] = results
        return df

    def shared_step(
        self,
        batch_dictionary,
        log_list=None,
    ):
        #Unpack the data from the batch dictionary
        Attention_Mask = batch_dictionary["Attention_Mask"]
        Batch_Mask = batch_dictionary["Batch_Mask"]
        Site_Feature = batch_dictionary["Site_Feature_Tensor"]
        Interaction_Features = batch_dictionary["Interaction_Feature_Tensor"]
        Atomic_ID = batch_dictionary["Atomic_ID"]
        Oxidation_State = batch_dictionary["Oxidation_State"]
        #Process Samples through input handler
        x = self.input_handler(Atomic_ID, [Site_Feature, Oxidation_State])
        # Pass through Encoder to get the global representation
        Global_Embedding = self.encoder.forward(
            x,
            Interaction_Features,
            Attention_Mask,
            Batch_Mask
        )
        #Perform the final layer and get the prediction
        prediction = af_dict[self.config["last_af_func"]](self.decoder(Global_Embedding))
        #Makes sure the prediction is a scalar just in case its a length 1 array
        while len(prediction.shape) > 1:
            prediction = prediction.squeeze()
        #Compute the MAE of the batch
        MAE = torch.abs(prediction - batch_dictionary["target"]).mean()
        #Log the average prediction
        prediction = prediction.mean()
        #Log for tensorboard
        if log_list is not None:
            for i in log_list:
                self.log(i[0], i[2](locals()[i[1]]), **i[3])
        return MAE
    #Makes sure the model is in training mode, passes a batch through the model, then back propogates
    def training_step(self, batch_dictionary, batch_dictionary_idx):
        self.train()
        log_list = [
            ["MAE", "MAE", lambda _: _, {}, {"prog_bar": True}],
            ["prediction", "prediction", lambda _: _, {}, {"prog_bar": True}],
        ]
        return self.shared_step(
            batch_dictionary,
            log_list=log_list,
        )
    #Makes sure the model is in eval mode then passes a validation sample through the model
    def validation_step(self, batch_dictionary, batch_dictionary_idx):
        self.eval()
        log_list = None
        return self.shared_step(batch_dictionary, log_list=log_list)

    #Configures the optimizer from the config
    def configure_optimizers(self):
        Optimizer_Config = self.config["Optimizer"]
        optimizer = optim_dict[Optimizer_Config["Name"]](
            self.parameters(),
            lr=self.config["Learning_Rate"],
            **Optimizer_Config["Kwargs"],
        )
        return {
            "optimizer": optimizer,
        }

    #Log the validation loss on every validation epoch
    def validation_epoch_end(self, outputs):
        self.avg_loss = torch.stack(outputs).mean()
        self.log("avg_val_loss", self.avg_loss)
        self.log("hp_metric", self.avg_loss)

class SiteNet_DIM(pl.LightningModule):
    def __init__(
        self,
        config=None,
    ):
        super().__init__()
        if config != None: #Implies laoding from a checkpoint if None
            #Load in the hyper parameters as lightning model attributes
            self.config = config
            self.batch_size = self.config["Batch_Size"]

            self.Site_DIM = SiteNetDIMAttentionBlock(**config)
            self.Global_DIM = SiteNetDIMGlobal(**config)
            self.Site_Prior = nn.Sequential(nn.Linear(config["attention_heads"]*config["site_dim_per_head"],256),nn.Mish(),nn.Linear(256,1))
            self.Global_Prior = nn.Sequential(nn.Linear(config["post_pool_layers"][-1],256),nn.Mish(),nn.Linear(256,1))
            self.decoder = nn.Sequential(nn.Linear(self.config["post_pool_layers"][-1], 256),nn.Mish(),nn.Linear(256, 1))

            self.config["pre_pool_layers_n"] = len(config["pre_pool_layers"])
            self.config["pre_pool_layers_size"] = sum(config["pre_pool_layers"]) / len(
                config["pre_pool_layers"]
            )
            self.config["post_pool_layers_n"] = len(config["post_pool_layers"])
            self.config["post_pool_layers_size"] = sum(config["post_pool_layers"]) / len(
                config["post_pool_layers"]
            )
            self.save_hyperparameters(self.config)
            self.automatic_optimization=False
    #Constructs the site features from the individual pieces, including the learnt atomic embeddings if enabled
    def input_handler(self, atomic_number, features):
        Atomic_Embedding = F.one_hot(atomic_number, num_classes=115).float()
        for i in features:
            assert not torch.isnan(i).any()
        if torch.isnan(Atomic_Embedding).any():
            print(atomic_number)
            print(Atomic_Embedding)
            raise (Exception)
        return torch.cat([Atomic_Embedding, *features], dim=1)

    #Inference mode, return the prediction
    def forward(self, b, batch_size=16,return_truth = False):
        self.eval()
        lob = [b[i : min(i + batch_size,len(b))] for i in range(0, len(b), batch_size)]
        Encoding_list = []
        targets_list= []
        print("Inference in batches of %s" % batch_size)
        for inference_batch in tqdm(lob):
            batch_dictionary = collate_fn(inference_batch, inference=True)
            Attention_Mask = batch_dictionary["Attention_Mask"]
            Site_Feature = batch_dictionary["Site_Feature_Tensor"]
            Atomic_ID = batch_dictionary["Atomic_ID"]
            Interaction_Features = batch_dictionary["Interaction_Feature_Tensor"]
            Oxidation_State = batch_dictionary["Oxidation_State"]
            concat_embedding = self.input_handler(
                Atomic_ID, [Site_Feature, Oxidation_State]
            )
            with torch.no_grad():
                Encoding = self.encoder.forward(
                    concat_embedding,
                    Interaction_Features,
                    Attention_Mask,
                    return_std=False,
                )
                Encoding = af_dict[self.config["last_af_func"]](self.decoder(Encoding))
                if self.config["regularization strategy"] == "l1_sparse":
                    Encoding = F.relu(Encoding)
                if self.config["regularization strategy"] == "kl_sparse":
                    Encoding = F.relu(Encoding)
                Encoding_list.append(Encoding)
                targets_list.append(batch_dictionary["target"])
        Encoding = torch.cat(Encoding_list, dim=0)
        targets = torch.cat(targets_list, dim=0)
        self.train()
        if return_truth:
            return [Encoding,targets]
        else:
            return Encoding

    #Feed in a dataframe with cif files in the "structure" column and get predictions
    def prepare_dataframe(
        self, df, structure_column_name="structure", input_dict_column_name="DIM_input"
    ):
        structures = df[structure_column_name]
        if self.config["Site_Features"] != None:
            Site_Featurizers = [
                site_feauturizers_dict[feat_dict["name"]](
                    *feat_dict["Featurizer_PArgs"], **feat_dict["Featurizer_KArgs"]
                )
                for feat_dict in self.config["Site_Features"]
            ]
        else:
            Site_Featurizers = None
        interaction_featurizers = [
            interaction_featurizer(
                interaction_featurizer_Functions[feat_dict["name"]], **feat_dict["kwargs"]
            )
            for feat_dict in self.config["Interaction_Features"]
        ]
        with multiprocessing.Pool(cpu_count() * 2) as pool:
            iterable = [[i, Site_Featurizers, interaction_featurizers] for i in structures]
            print("calculating site features")
            results = list(
                tqdm(
                    pool.istarmap(
                        structure_processing_site_features,
                        iterable,
                    ),
                    total=len(iterable),
                )
            )
            print("Reading matricies / elemental data from structures")
            results = [
                structure_processing_static(i, j)
                for i, j in tqdm(zip(structures, results))
            ]

        df[input_dict_column_name] = results
        return df

    #Makes sure the model is in training mode, passes a batch through the model, then back propogates
    def training_step(self, batch_dictionary, batch_dictionary_idx):
        self.train()
        local_opt,global_opt,task_opt,local_prior_opt,global_prior_opt = self.optimizers()
        Attention_Mask = batch_dictionary["Attention_Mask"]
        Batch_Mask = batch_dictionary["Batch_Mask"]
        Site_Features = batch_dictionary["Site_Feature_Tensor"]
        Interaction_Features = batch_dictionary["Interaction_Feature_Tensor"]
        Atomic_ID = batch_dictionary["Atomic_ID"]
        Oxidation_State = batch_dictionary["Oxidation_State"]
        #Process Samples through input handler
        Site_Features = self.input_handler(Atomic_ID, [Site_Features, Oxidation_State])

        #Perform a step on creating local environment representations while tricking the prior discriminator
        local_opt.zero_grad()
        Local_Environment_Features,Local_Environment_Loss,Local_Environment_DIM_loss,Local_Environment_KL_loss = self.Site_DIM(Site_Features, Interaction_Features, Attention_Mask, Batch_Mask)
        Local_prior_samples = torch.randn_like(Local_Environment_Features)
        Local_prior_score = F.softplus(-self.Site_Prior(Local_prior_samples))
        Local_posterior_score = F.softplus(self.Site_Prior(Local_Environment_Features))
        #Get prior loss per site
        Local_prior_loss = Local_prior_score+Local_posterior_score
        #Get prior loss per crystal
        Local_prior_loss = segment_csr(Local_prior_loss,Batch_Mask["CSR"],reduce="mean")
        #Get prior loss per batch
        Local_prior_loss = Local_prior_loss.flatten().mean()
        Local_Environment_Loss = Local_Environment_Loss + 0.2*Local_prior_loss
        #self.manual_backward(Local_Environment_Loss)
        #local_opt.step()

        #Adversarially train the prior discriminator
        local_prior_opt.zero_grad()
        Local_prior_score = F.softplus(self.Site_Prior(Local_prior_samples))
        Local_posterior_score = F.softplus(-self.Site_Prior(Local_Environment_Features.detach().clone()))
        #Get prior loss per site
        Site_prior_loss = Local_prior_score+Local_posterior_score
        #Get prior loss per crystal
        Site_prior_loss = segment_csr(Site_prior_loss,Batch_Mask["CSR"],reduce="mean")
        #Get prior loss per batch
        Site_prior_loss = Site_prior_loss.flatten().mean()
        #self.manual_backward(Site_prior_loss)
        #local_prior_opt.step()

        #Perform a step on creating global environment representations, loss depends on mutual information and being able to trick the prior discriminator
        global_opt.zero_grad()
        Global_Embedding_Features,Global_Loss,Global_DIM_loss,Global_KL_loss = self.Global_DIM(Local_Environment_Features.detach().clone(),Batch_Mask)
        Global_prior_samples = torch.randn_like(Global_Embedding_Features)
        Global_prior_score = F.softplus(-self.Global_Prior(Global_prior_samples))
        Global_posterior_score = F.softplus(self.Global_Prior(Global_Embedding_Features))
        Global_prior_loss = (Global_prior_score+Global_posterior_score).flatten().mean()
        Global_Loss = Global_Loss + 0.2*Global_prior_loss
        #self.manual_backward(Global_Loss)
        #global_opt.step()

        #Train the prior discriminator
        global_prior_opt.zero_grad()
        Global_prior_score = F.softplus(self.Global_Prior(Global_prior_samples))
        Global_posterior_score = F.softplus(-self.Global_Prior(Global_Embedding_Features.detach().clone()))
        Global_prior_loss = (Global_prior_score+Global_posterior_score).flatten().mean()
        #self.manual_backward(Global_prior_loss)
        #global_prior_opt.step()

        #Perform a step on predicting the band gap with the learnt global embedding
        task_opt.zero_grad()
        Prediction = self.decoder(Global_Embedding_Features.detach().clone())
        MAE = torch.abs(Prediction - batch_dictionary["target"]).mean()
        self.manual_backward(MAE)
        task_opt.step()

        #"Local_Environment_KL_loss":Local_Environment_KL_loss,
        #"Global_KL_loss":Global_KL_loss,

        self.log_dict({"Local_Loss":Local_Environment_Loss,"Global_Loss":Global_Loss,"task_loss":MAE,"Local_Environment_DIM_Loss":Local_Environment_DIM_loss,
        "Global_DIM_loss":Global_DIM_loss,"Local_prior_loss":Site_prior_loss,"Global_prior_loss":Global_prior_loss},prog_bar=True)
    #Makes sure the model is in eval mode then passes a validation sample through the model
    def validation_step(self, batch_dictionary, batch_dictionary_idx):
        self.eval()
        Attention_Mask = batch_dictionary["Attention_Mask"]
        Batch_Mask = batch_dictionary["Batch_Mask"]
        Site_Features = batch_dictionary["Site_Feature_Tensor"]
        Interaction_Features = batch_dictionary["Interaction_Feature_Tensor"]
        Atomic_ID = batch_dictionary["Atomic_ID"]
        Oxidation_State = batch_dictionary["Oxidation_State"]
        #Process Samples through input handler
        Site_Features = self.input_handler(Atomic_ID, [Site_Features, Oxidation_State])
        #Perform site deep infomax to obtain loss and embedding
        Local_Environment_Features,Local_Environment_Loss,Local_Environment_DIM_loss,Local_Environment_KL_loss = self.Site_DIM(Site_Features, Interaction_Features, Attention_Mask, Batch_Mask)
        #Detach the local nevironment features and do independant deep infomax to convert local environment features to global features
        Global_Embedding_Features,Global_Loss,Global_DIM_loss,Global_KL_loss = self.Global_DIM(Local_Environment_Features,Batch_Mask)
        #Try and perform shallow property prediction using the global representation as a sanity check
        Prediction = self.decoder(Global_Embedding_Features)
        MAE = torch.abs(Prediction - batch_dictionary["target"]).mean()
        return [Local_Environment_Loss,Global_Loss,MAE,Local_Environment_DIM_loss,Local_Environment_KL_loss,Global_DIM_loss,Global_KL_loss]

    #Configures the optimizer from the config
    def configure_optimizers(self):
        Optimizer_Config = self.config["Optimizer"]
        #Local DIM optimizer
        local_opt = optim_dict[Optimizer_Config["Name"]](
            self.Site_DIM.parameters(),
            lr=self.config["Learning_Rate"],
            **Optimizer_Config["Kwargs"],)
        #Global DIM optimizer
        global_opt = optim_dict[Optimizer_Config["Name"]](
            self.Global_DIM.parameters(),
            lr=self.config["Learning_Rate"],
            **Optimizer_Config["Kwargs"],)
        #Task optimizer
        task_opt = optim_dict[Optimizer_Config["Name"]](
            self.decoder.parameters(),
            lr=self.config["Learning_Rate"],
            **Optimizer_Config["Kwargs"],)
        #Local prior optimizer
        local_prior_opt = optim_dict[Optimizer_Config["Name"]](
            self.Site_Prior.parameters(),
            lr=self.config["Learning_Rate"],
            **Optimizer_Config["Kwargs"],)
        #Global prior optimizer
        global_prior_opt = optim_dict[Optimizer_Config["Name"]](
            self.Global_Prior.parameters(),
            lr=self.config["Learning_Rate"],
            **Optimizer_Config["Kwargs"],)

        return local_opt,global_opt,task_opt,local_prior_opt,global_prior_opt

    #Log the validation loss on every validation epoch
    def validation_epoch_end(self, outputs):
        self.avg_loss_local = torch.stack([i[0] for i in outputs]).mean()
        self.avg_loss_global = torch.stack([i[1] for i in outputs]).mean()
        self.avg_loss_task = torch.stack([i[2] for i in outputs]).mean()
        self.Local_Environment_DIM_loss = torch.stack([i[3] for i in outputs]).mean()
        #self.Local_Environment_KL_loss = torch.stack([i[4] for i in outputs]).mean()
        self.Global_DIM_loss = torch.stack([i[5] for i in outputs]).mean()
        #self.Global_KL_loss = torch.stack([i[6] for i in outputs]).mean()
        self.log("avg_val_loss_local", self.avg_loss_local)
        self.log("avg_val_loss_global", self.avg_loss_global)
        self.log("avg_val_loss_task", self.avg_loss_task)
        self.log("avg_val_loss_local_DIM",self.Local_Environment_DIM_loss)
        self.log("avg_val_loss_global_DIM",self.Global_DIM_loss)
        #self.log("avg_val_loss_local_KL",self.Local_Environment_KL_loss)
        #self.log("avg_val_loss_global_KL",self.Global_KL_loss)

from pymatgen.transformations.standard_transformations import *

class basic_callbacks(pl.Callback):
    def __init__(self,*pargs,filename = "current_model",**kwargs):
        super().__init__(*pargs,**kwargs)
        self.filename = filename + ".ckpt"
    
    def on_init_start(self, trainer):
        print("Starting to init trainer!")

    def on_init_end(self, trainer):
        print("trainer is init now")

    def on_train_end(self, trainer, model):
        trainer.save_checkpoint("most_recent_complete_run.ckpt")

    def on_train_epoch_end(self, trainer, pl_module):
        trainer.save_checkpoint("current_model.ckpt")

    def on_validation_epoch_end(self, trainer, pl_module):
        trainer.save_checkpoint(self.filename)


############ DATA MODULE ###############

class DIM_h5_Data_Module(pl.LightningDataModule):
    def __init__(
        self,
        config,
        overwrite=False,
        ignore_errors=False,
        chunk_size=cpu_count() ** 2 * 16,
        max_len=100,
        Dataset=None,
    ):

        super().__init__()
        self.batch_size = config["Batch_Size"]
        #In dynamic batching, the number of unique sites is the limit on the batch, not the number of crystals, the number of crystals varies between batches
        self.dynamic_batch = config["dynamic_batch"]
        self.Site_Features = config["Site_Features"]
        self.Interaction_Features = config["Interaction_Features"]
        self.h5_file = config["h5_file"]
        self.overwrite = overwrite
        self.ignore_errors = ignore_errors
        self.limit = config["Max_Samples"]
        self.chunk_size = chunk_size
        self.max_len = max_len
        if Dataset is None:
            self.Dataset = torch_h5_cached_loader(
                self.Site_Features,
                self.Interaction_Features,
                self.h5_file,
                max_len=self.max_len,
                ignore_errors=self.ignore_errors,
                overwrite=self.overwrite,
                limit=self.limit,
            )
        else:
            self.Dataset = Dataset

    def prepare_data(self):
        self.Dataset_Train, self.Dataset_Val = random_split(
            self.Dataset,
            [len(self.Dataset) - len(self.Dataset) // 20, len(self.Dataset) // 20],
            generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self):
        if self.dynamic_batch:
            return DataLoader(
                self.Dataset_Train,
                collate_fn=collate_fn,
                batch_sampler=SiteNet_batch_sampler(RandomSampler(self.Dataset_Train),self.batch_size),
                pin_memory=False,
                num_workers=cpu_count()-1,
                prefetch_factor=8,
                persistent_workers=True
            )
        else:
            return DataLoader(
                self.Dataset_Train,
                batch_size=self.batch_size,
                collate_fn=collate_fn,
                pin_memory=False,
                num_workers=cpu_count()-1,
                prefetch_factor=8,
                persistent_workers=True
            )

    def val_dataloader(self):
        if self.dynamic_batch:
            return DataLoader(
                self.Dataset_Val,
                collate_fn=collate_fn,
                batch_sampler=SiteNet_batch_sampler(RandomSampler(self.Dataset_Val),self.batch_size),
                pin_memory=False,
                num_workers=cpu_count()-1,
                prefetch_factor=8,
                persistent_workers=True
            )
        else:
            return DataLoader(
                self.Dataset_Val,
                batch_size=self.batch_size,
                collate_fn=collate_fn,
                pin_memory=False,
                num_workers=cpu_count()-1,
                prefetch_factor=8,
                persistent_workers=True
            )
class SiteNet_batch_sampler(Sampler):
    def __init__(self, sampler, batch_size):
        self.sampler = sampler
        self.batch_size = batch_size
        print("Initializing Random Sampler")
        self.prim_sizes = {idx:self.sampler.data_source[idx]["prim_size"] for idx in tqdm(sampler)}
    def __iter__(self):
            #The VRAM required by the model in each batch is proportional to the number of unique atomic sites, or the length of the i axis
            #Keep extending the batch with more crystals until the batch limit is exceeded
            #Batch limit will be exceeded by at most 1 crystal to avoid throwing away indicies
            sampler_iter = iter(self.sampler)
            batch = []
            size = 0
            idx = next(sampler_iter)
            while True:
                try:
                    #Check if this crystal brings us above the batch limit
                    if self.prim_sizes[idx] + size > self.batch_size:
                        yield batch
                        size = self.prim_sizes[idx]
                        batch = [idx]
                        idx = next(sampler_iter)
                    else:
                        size+= self.prim_sizes[idx]
                        batch.append(idx)
                        idx = next(sampler_iter)
                except StopIteration:
                    #Don't throw away the last batch if it isnt empty
                    if batch != []:
                        yield batch
                    #break to let lightning know the epoch is over
                    break

if __name__ == "__main__":
    import json

    config = json.load(open("config/test.json", "rb"))
    model = Attention_Infomax(config)
    train_loader = model.train_dataloader()
    model.training_step(model.Dataset[0], 0)
    for _ in train_loader:
        pass

