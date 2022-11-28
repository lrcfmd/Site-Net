import torch
import pytorch_lightning as pl
from h5_handler import *
from torch.utils.data import DataLoader, random_split
from multiprocessing import cpu_count
from torch.nn.functional import pad
from modules import (
    SiteNetEncoder
)
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.optim.lr_scheduler import *
from torch import nn
from torch_scatter import segment_csr
from torch.utils.data import RandomSampler,Sampler
from pymatgen.transformations.standard_transformations import *

#Clamps negative predictions to zero without interfering with the gradients. "Transparent" ReLU
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

#Dictionaries allow programatic access of torch modules according to the config file
optim_dict = torch.optim.__dict__
site_feauturizers_dict = matminer.featurizers.site.__dict__
af_dict = {"identity":lambda x:x,"relu":nn.functional.relu,"softplus":nn.functional.softplus,"TReLU":TReLU.apply}


#Constructs a batch dictionary from the list of property dictionaries returned by the h5 loader
#Also performs necessary zero padding on the j axis and creates the batch masks
def collate_fn(batch, inference=False):
    batch_dict = {}
    #Necessary information to perform the batching
    primitive_lengths = [i["prim_size"] for i in batch]
    image_count = [i["images"] for i in batch]
    actual_length = [i["prim_size"]*i["images"] for i in batch]

    #Turns the numpy arrays into torch tensors ready for the model
    def initialize_tensors(batch_full, key, dtype, process_func=lambda _: _):
        batch = [i[key] for i in batch_full]
        batch = [process_func(i) for i in batch]
        batch = [torch.as_tensor(np.array(i), dtype=dtype) for i in batch]
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
    Atomic_ID = initialize_tensors(batch, "Atomic_ID", torch.long)
    site_features = initialize_tensors(batch, "Site_Feature_Tensor", torch.float)
    interaction_features = initialize_tensors(batch, "Interaction_Feature_Tensor", torch.float)
    Oxidation_State = initialize_tensors(batch, "Oxidation_State", torch.float)
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
                Encoding_list.append(Encoding)
                targets_list.append(batch_dictionary["target"])
        Encoding = torch.cat(Encoding_list, dim=0)
        targets = torch.cat(targets_list, dim=0)
        self.train()
        if return_truth:
            return [Encoding,targets]
        else:
            return Encoding

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

class basic_callbacks(pl.Callback):
    def __init__(self,*pargs,filename = "current_model",**kwargs):
        super().__init__(*pargs,**kwargs)
        self.filename = filename + ".ckpt"

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
        max_len=100,
        Dataset=None,
        cpus = 1,
        chunk_size = 32,
        **kwargs
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
        self.max_len = max_len
        self.cpus=cpus
        if Dataset is None:
            self.Dataset = torch_h5_cached_loader(
                self.Site_Features,
                self.Interaction_Features,
                self.h5_file,
                max_len=self.max_len,
                ignore_errors=self.ignore_errors,
                overwrite=self.overwrite,
                limit=self.limit,
                cpus=cpus,
                chunk_size=chunk_size
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
                num_workers=self.cpus,
                prefetch_factor=8,
                persistent_workers=True
            )
        else:
            return DataLoader(
                self.Dataset_Train,
                batch_size=self.batch_size,
                collate_fn=collate_fn,
                pin_memory=False,
                num_workers=self.cpus,
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
                num_workers=self.cpus,
                prefetch_factor=8,
                persistent_workers=True
            )
        else:
            return DataLoader(
                self.Dataset_Val,
                batch_size=self.batch_size,
                collate_fn=collate_fn,
                pin_memory=False,
                num_workers=self.cpus,
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
            #Keep extending the batch with more crystals until doing so again brings us over the batch size
            #If extending the batch would bring it over the max batch size, yield the current batch and seed a new one
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
