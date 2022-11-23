# Training data are stored in an h5 file on a per cif basis
# Featurization of training data is done JIT (Just in time) by either loading the featurized tensor from the h5 file or generating and writing if not
# This is more of a cache than a database, means you only have to generate the features for the crystals once
# WARNING: h5 files are corrupted if unexpectedly closed during a write operation

from torch.utils.data import Dataset
import h5py
import numpy as np
import matminer
import matminer.featurizers.site as site
import pickle as pk
from dscribe.descriptors import SOAP as SOAP_dscribe
import traceback
import pickle as pk
from compress_pickle import dumps, loads
import traceback
import multiprocessing
import multiprocessing.pool as mpp
from multiprocessing import cpu_count,Process, Pool, set_start_method
from random import shuffle, seed
seed(42)
import yaml
from pytorch_lightning.callbacks import *
import argparse
import resource
import matminer.featurizers.structure as structure_feat
import random
from torch import unsqueeze
from pymatgen.analysis.local_env import *
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

site_feauturizers_dict = matminer.featurizers.site.__dict__


def istarmap(self, func, iterable, chunksize=1):
    """starmap-version of imap"""
    self._check_running()
    if chunksize < 1:
        raise ValueError("Chunksize must be 1+, not {0:n}".format(chunksize))

    task_batches = mpp.Pool._get_tasks(func, iterable, chunksize)
    result = mpp.IMapIterator(self)
    self._taskqueue.put(
        (
            self._guarded_task_generation(result._job, mpp.starmapstar, task_batches),
            result._set_length,
        )
    )
    return (item for chunk in result for item in chunk)


mpp.Pool.istarmap = istarmap

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))
set_start_method("fork")
# torch.multiprocessing.set_sharing_strategy("file_system")

site_feauturizers_dict = matminer.featurizers.site.__dict__
comp_alg = "gzip"


# Class to convert bond feature functions into featurizer classes


class bond_featurizer:
    def __init__(self, base_function, log=False, polynomial_degree=1, max_clip=10000):
        self.base_function = base_function
        self.polynomial = polynomial_degree
        self.log = log
        self.max_clip = max_clip

    def featurize(self, structure):
        base_matrix = self.base_function(structure)
        base_matrix = base_matrix**self.polynomial
        if self.log:
            base_matrix = np.log(base_matrix + 1e-8)
            base_matrix = np.clip(base_matrix, -self.max_clip, self.max_clip)
        return base_matrix


# Bond featurizer functions that except to recieve a pymatgen structure and return a 3d adjaceny matrix of dimension NxMxF
def distance_matrix(structure, func=lambda _: _):
    distance_matrix = func(structure.distance_matrix)
    return distance_matrix


def sine_coulomb_matrix(structure):
    return structure_feat.SineCoulombMatrix(diag_elems=True, flatten=False).featurize(
        structure
    )[0]


def coulomb_matrix(structure):
    return structure_feat.CoulombMatrix(diag_elems=True, flatten=False).featurize(
        structure
    )[0]


# Dictionary of Bond Featurizers
Bond_Featurizer_Functions = {
    "distance_matrix": distance_matrix,
    "reciprocal_square_distance_matrix": lambda structure_l: distance_matrix(
        structure_l, func=lambda _: _**-2
    ),
    "coulomb_matrix": sine_coulomb_matrix,
    "non_sine_coulomb_matrix": coulomb_matrix,
}


def clean_results(result):
    for idx in range(len(result))[::-1]:
        if result[idx] == "Invalid":
            del result[idx]
        else:
            result[idx] = dumps(result[idx], comp_alg)


def clean_result(i, key_list, max_len):
    valid = True
    for key in key_list:
        if i[key] is None:
            return "Invalid"
        else:
            if np.isnan(i[key]).any():
                return "Invalid"
    if max_len is not None and valid == True:
        if i["Atomic_ID"].shape[0] > max_len:
            return "Invalid"
    return i


class site_agnostic_SOAP(site.SOAP):
    def featurize(self, struct, idx):
        featurized = np.array(super().featurize(struct, idx))
        featurized = np.array(np.split(featurized, 100))
        featurized = np.concatenate(
            [
                np.sum(featurized, axis=0),
                np.std(featurized, axis=0),
                np.max(featurized, axis=0),
            ]
        )
        return featurized


def agnostic_SOAP_Wrapper(*pargs, average="off", **kwargs):
    soap = site_agnostic_SOAP(*pargs, **kwargs)
    soap.soap = SOAP_dscribe(
        soap.rcut,
        soap.nmax,
        soap.lmax,
        sigma=soap.sigma,
        species=[i + 1 for i in range(100)],
        rbf=soap.rbf,
        periodic=soap.periodic,
        crossover=soap.crossover,
        average=average,
        sparse=False,
    )
    return soap


def SOAP_Wrapper(*pargs, average="off", **kwargs):
    soap = site.SOAP(*pargs, **kwargs)
    soap.soap = SOAP_dscribe(
        soap.rcut,
        soap.nmax,
        soap.lmax,
        sigma=soap.sigma,
        species=[i + 1 for i in range(100)],
        rbf=soap.rbf,
        periodic=soap.periodic,
        crossover=soap.crossover,
        average=average,
        sparse=False,
    )
    return soap


site_feauturizers_dict["SOAP_dscribe"] = SOAP_Wrapper
site_feauturizers_dict["agnostic_SOAP_dscribe"] = agnostic_SOAP_Wrapper

SCM = structure_feat.SineCoulombMatrix(
    flatten=False,
)

# Helper classes & functions
def list_to_dict(list_of_dicts, list_of_values, key):
    for i, j in zip(list_of_dicts, list_of_values):
        i[key] = j
    return list_of_dicts


class InvalidEntry(Exception):
    pass


# Main Block


class Writer(Process):
    def __init__(self, task_queue, fname):
        super().__init__()
        # multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self._fname = fname

    def run(self):
        try:
            self.f = h5py.File(self._fname, "r+")
            while True:
                next_task = self.task_queue.get()
                if next_task is None:
                    # Poison pill means shutdown
                    # print("writer: Exiting")
                    self.f.flush()
                    self.f.close()
                    self.task_queue.task_done()
                    break
                # print('writer: %s' % (next_task["Group_Name"]))
                try:
                    if np.ndim(next_task["Feature_Array"]) > 0:
                        self.f[next_task["Group_Name"]].create_dataset(
                            next_task["Name"],
                            data=next_task["Feature_Array"],
                            compression="gzip",
                        )
                    else:
                        self.f[next_task["Group_Name"]].create_dataset(
                            next_task["Name"],
                            data=next_task["Feature_Array"],
                        )
                    # print("write")
                except KeyboardInterrupt as e:
                    pass
                except Exception as e:
                    self.f[next_task["Group_Name"]][next_task["Name"]][()] = next_task[
                        "Feature_Array"
                    ]
                    print("overwrite")
                self.task_queue.task_done()
        except KeyboardInterrupt as k:
            pass
        except Exception as e:
            traceback.print_exc()
            raise (e)


def read_structures(h5_group, h5_file):
    try:
        structure = pk.loads(h5_file[h5_group]["pymatgen_structure"][()])
        target = h5_file[h5_group]["target"][()]
        prim_size = h5_file[h5_group]["prim_size"][()]
        images = h5_file[h5_group]["images"][()]
        return structure, target, prim_size, images
    except KeyboardInterrupt as e:
        raise e
    except Exception as e:
        traceback.print_exc()
        print("Loading Structure Failed " + str(h5_group))
        return None


def populate_h5_state_dict(preprocessing_dict_list, file, h5_group, ignore_errors):
    state_dict = {}
    if preprocessing_dict_list is not None:
        for pre in preprocessing_dict_list:
            try:
                loaded_value = file[h5_group][str(pre)][()]
                if np.isscalar(loaded_value) and not np.any(
                    loaded_value,
                ):
                    if ignore_errors:
                        state_dict[str(pre)] = "Generate"
                    else:
                        print("Failed")
                        state_dict[str(pre)] = "Failed"
                else:
                    state_dict[str(pre)] = loaded_value
            except KeyboardInterrupt as e:
                raise e
            except Exception as e:
                state_dict[str(pre)] = "Generate"
    return state_dict


def get_site_feature_array(pre, structure, prim_size, images):
    featurizer = site_feauturizers_dict[pre["name"]](
        *pre["Featurizer_PArgs"], **pre["Featurizer_KArgs"]
    )
    feature_array = np.array(
        [
            featurizer.featurize(structure, i) for i in range(0, len(structure), images)
        ]  # Skip over identical sites with linspacing equal to the number of images
    )
    return feature_array


def get_bonds_feature_array(pre, structure, prim_size, images):
    featurizer = bond_featurizer(
        Bond_Featurizer_Functions[pre["name"]], **pre["kwargs"]
    )
    # Generating a supercell puts all identical sites next to eachother, linear spacing equal to the number of images obtains the unique sites
    feature_array = np.array(featurizer.featurize(structure))[
        0 : len(structure) : images
    ]
    return feature_array


def generate_site(
    pre, structure, task_queue, h5_group, feature_arrays, prim_size, images
):
    feature_array = get_site_feature_array(pre, structure, prim_size, images)
    task_queue.put(
        {"Name": str(pre), "Feature_Array": feature_array, "Group_Name": h5_group}
    )
    feature_arrays.append(feature_array)


def generate_bonds(
    pre, structure, task_queue, h5_group, feature_arrays, prim_size, images
):
    feature_array = get_bonds_feature_array(pre, structure, prim_size, images)
    task_queue.put(
        {"Name": str(pre), "Feature_Array": feature_array, "Group_Name": h5_group}
    )
    feature_arrays.append(feature_array)


def poison_pill(pre, task_queue, h5_group, feature_arrays):
    task_queue.put(
        {"Name": str(pre), "Feature_Array": np.array(0), "Group_Name": h5_group}
    )
    feature_arrays.append("Poison Pill")


def featurize_h5_cache_site_features(
    task_queue,
    preprocessing_dict_list,
    bond_preprocessing_dict_list,
    structure_args,
    read_args_site,
    read_args_bond,
    h5_group,
    overwrite,
    ignore_errors,
):
    structure, target, prim_size, images = read_structures(*structure_args)
    read_site = populate_h5_state_dict(*read_args_site)
    read_bond = populate_h5_state_dict(*read_args_bond)
    if structure is None:
        return None

    # Try block for loading site features
    try:
        feature_arrays = []
        if preprocessing_dict_list != None:
            for pre in preprocessing_dict_list:
                if overwrite:
                    generate_site(pre, structure, task_queue, h5_group, feature_arrays)
                else:
                    loaded_value = read_site[str(pre)]
                    if type(loaded_value) == str:
                        if loaded_value == "Failed" and ignore_errors is not True:
                            return None
                        elif loaded_value == "Generate":
                            try:
                                generate_site(
                                    pre,
                                    structure,
                                    task_queue,
                                    h5_group,
                                    feature_arrays,
                                    prim_size,
                                    images,
                                )
                            except KeyboardInterrupt as e:
                                raise e
                            except Exception as e:
                                # traceback.print_exc()
                                print(e)
                                poison_pill(pre, task_queue, h5_group, feature_arrays)
                        else:
                            raise InvalidEntry(
                                'Load dictionary should either be the value, "Failed" or "Generate"'
                            )
                    else:
                        feature_arrays.append(loaded_value)
            site_feature_arrays = np.concatenate([i for i in feature_arrays], axis=1)
        else:
            site_feature_arrays = np.empty((len(structure), 0))
    except (InvalidEntry, KeyboardInterrupt) as e:
        raise e
    except Exception as e:
        traceback.print_exc()
        return None, None, structure, target

    # Try block for loading bond features
    try:
        feature_arrays = []
        for pre in bond_preprocessing_dict_list:
            if overwrite:
                generate_bonds(pre, structure, task_queue, h5_group, feature_arrays)
            else:
                loaded_value = read_bond[str(pre)]
                if type(loaded_value) == str:
                    if loaded_value == "Failed" and ignore_errors is not True:
                        return None
                    elif loaded_value == "Generate":
                        try:
                            generate_bonds(
                                pre,
                                structure,
                                task_queue,
                                h5_group,
                                feature_arrays,
                                prim_size,
                                images,
                            )
                        except KeyboardInterrupt as e:
                            raise e
                        except Exception as e:
                            traceback.print_exc()
                            print(e)
                            poison_pill(pre, task_queue, h5_group, feature_arrays)
                    else:
                        raise InvalidEntry(
                            'Load dictionary should either be the value, "Failed" or "Generate"'
                        )
                else:
                    feature_arrays.append(loaded_value)
        for i in range(len(feature_arrays)):
            while feature_arrays[i].ndim < 3:
                feature_arrays[i] = np.expand_dims(
                    feature_arrays[i], feature_arrays[i].ndim
                )
        bond_feature_arrays = np.concatenate([i for i in feature_arrays], axis=2)
    except (InvalidEntry, KeyboardInterrupt) as e:
        raise e
    except Exception as e:
        # traceback.print_exc()
        return None, None, structure, target
    return (
        site_feature_arrays,
        bond_feature_arrays,
        structure,
        target,
        prim_size,
        images,
    )


# Token Loader


def featurize_h5_cache_Oxidation(structure, images):
    try:
        oxidation_list = []
        for idx, i in enumerate(structure):
            # Skip over identical sites using linear spacing equal to the number of images
            if idx in list(range(0, len(structure), images)):
                try:
                    oxidation_list.append(np.array([i.specie.oxi_state]))
                except:
                    oxidation_list.append(np.array([0]))
        oxi_list = oxidation_list
        return oxi_list
    except KeyboardInterrupt as e:
        raise KeyboardInterrupt
    except Exception as e:
        traceback.print_exc()
        print(e)
        return None


def featurize_h5_cache_ElemToken(structure, images):
    try:
        # Generating a supercell puts all identical sites next to eachother, linear spacing equal to the number of images obtains the unique sites
        token_list = np.array([int(i.specie.Z) for i in structure])[
            0 : len(structure) : images
        ]
        mask_high = token_list >= 100
        mask_low = token_list <= 0
        mask = mask_high | mask_low
        if mask.any():
            raise (Exception)
        return token_list
    except KeyboardInterrupt as e:
        raise KeyboardInterrupt
    except Exception as e:
        traceback.print_exc()
        print(e)
        return None


def result_get(
    keys,
    site_features_config,
    bond_features_config,
    h5_file_name,
    overwrite,
    ignore_errors,
    tasks,
    max_len,
):
    with h5py.File(h5_file_name, "r") as h5_file:
        result = [
            {"ICSD": i} for i in keys
        ]  # ID is no longer tied to the ICSD, this is a vestigal name for backwards compatability with old datasets
        # Reading
        del h5_file_name
        structure_args = ((key, h5_file) for key in keys)
        Read_values_dict_args = (
            (site_features_config, h5_file, key, ignore_errors) for key in keys
        )
        Read_values_dict_args_bonds = (
            (bond_features_config, h5_file, key, ignore_errors) for key in keys
        )
        # create queue and manager
        # Compute features and add to write queue
        # Site Features
        processed_structure_list = [
            [i, j, k, l, m, n]
            for i, j, k, l, m, n in [
                featurize_h5_cache_site_features(
                    tasks,
                    site_features_config,
                    bond_features_config,
                    i,
                    j1,
                    j2,
                    k,
                    overwrite,
                    ignore_errors,
                )
                for i, j1, j2, k in zip(
                    structure_args,
                    Read_values_dict_args,
                    Read_values_dict_args_bonds,
                    keys,
                )
            ]
        ]
        site_result = [i[0] for i in processed_structure_list]
        bond_result = [i[1] for i in processed_structure_list]
        structures = [i[2] for i in processed_structure_list]
        targets = [i[3] for i in processed_structure_list]
        prim_sizes = [i[4] for i in processed_structure_list]
        images = [i[5] for i in processed_structure_list]
        del processed_structure_list
        for local_dict, value in zip(result, site_result):
            local_dict["Site_Feature_Tensor"] = value
        for local_dict, value in zip(result, bond_result):
            local_dict["Interaction_Feature_Tensor"] = value
        # Load in Elemental Tokens
        Atomic_ID_List = [
            featurize_h5_cache_ElemToken(i, j) for i, j in zip(structures, images)
        ]
        Oxidation_List = [
            featurize_h5_cache_Oxidation(i, j) for i, j in zip(structures, images)
        ]
        result = list_to_dict(result, Atomic_ID_List, "Atomic_ID")
        result = list_to_dict(result, Oxidation_List, "Oxidation_State")
        result = list_to_dict(result, structures, "structure")
        result = list_to_dict(result, targets, "target")
        result = list_to_dict(result, prim_sizes, "prim_size")
        result = list_to_dict(result, images, "images")
        h5_file.close()
        keys = [
            "Site_Feature_Tensor",
            "Interaction_Feature_Tensor",
            "Atomic_ID",
            "Oxidation_State",
            "target",
            "prim_size",
            "images",
        ]
        # print(site_result)
        result = [i for i in [clean_result(i, keys, max_len) for i in result]]
        clean_results(result)
        return result


def n_list_chunks(lst, n):
    """Yield successive n-sized chunks from a list (lst)."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def JIT_h5_load(
    site_features_config,
    bond_features_config,
    h5_file_name,
    max_len,
    overwrite=False,
    ignore_errors=False,
    chunk_size=32,
    cpus=1,
    limit=None,
):
    print("h5 file name is " + h5_file_name)
    key_data = h5py.File(h5_file_name, "r")
    keys = list(key_data.keys())
    shuffle(keys)
    keys = keys[:limit]
    key_data.close()

    def divide_chunks(l, n):
        for i in range(0, len(l), n):
            yield l[i : i + n]

    keys_list = list(divide_chunks(keys, chunk_size*cpus))
    results = []
    print("Initializing data from h5 file in size " + str(chunk_size*cpus) + " Chunks")
    print("Worker process count is " + str(cpus))
    for keys in tqdm(keys_list):
        with Pool(cpus) as pool:
            m = multiprocessing.Manager()
            tasks = m.JoinableQueue()
            keys_chunk_list = n_list_chunks(keys, (max(1,len(keys) // cpus)))
            result_chunk = pool.starmap(
                result_get,
                [
                    [
                        keys_chunk,
                        site_features_config,
                        bond_features_config,
                        h5_file_name,
                        overwrite,
                        ignore_errors,
                        tasks,
                        max_len,
                    ]
                    for keys_chunk in keys_chunk_list
                ],
            )
        #serialise the inputs
        for chunk in result_chunk:
            results.extend(chunk)
        #If anything had to be computed write it to the file
        if tasks.qsize() > 0:
            print("Writing do not terminate process")
            print("Queue size is " + str(tasks.qsize()))
            tasks.put(None)
            writer = Writer(tasks, h5_file_name)
            writer.start()
            tasks.join()
            writer.join()
            writer.close()
            print("Writing complete, safe to terminate")
        #Kill any zombie workers
        multiprocessing.active_children()
    return results


class torch_h5_cached_loader(Dataset):
    @staticmethod
    def rand_false(idx, it_length):
        false_idx = random.randrange(it_length - 1)
        false_idx = false_idx + 1 if false_idx >= idx else false_idx
        return false_idx

    def __init__(
        self,
        Site_Features,
        Bond_Features,
        h5_file,
        overwrite=False,
        ignore_errors=False,
        limit=None,
        chunk_size=32,
        cpus = 1,
        max_len=None,
    ):
        self.chunk_size = chunk_size
        self.max_len = max_len
        self.result = JIT_h5_load(
            Site_Features,
            Bond_Features,
            h5_file,
            max_len,
            overwrite=overwrite,
            ignore_errors=ignore_errors,
            limit=limit,
            chunk_size=chunk_size,
            cpus=cpus
        )

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in list(range(idx.start, idx.stop))]
        Requested_Structure = loads(self.result[idx], comp_alg)
        False_Structure = loads(
            self.result[self.rand_false(idx, len(self.result))], comp_alg
        )
        Requested_Structure["False_Sample"] = {
            "Site_Feature_Tensor": False_Structure["Site_Feature_Tensor"],
            "Atomic_ID": False_Structure["Atomic_ID"],
            "Oxidation_State": False_Structure["Oxidation_State"],
            "Interaction_Feature_Tensor": False_Structure["Interaction_Feature_Tensor"],
        }
        return Requested_Structure

    def __len__(self):
        return len(self.result)


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ml options")
    parser.add_argument("-c", "--config", default="test")
    parser.add_argument("-p", "--pickle", default=0)
    parser.add_argument("-l", "--load_checkpoint", default=0)
    args = parser.parse_args()
    try:
        print(args.config)
        with open(str("config/" + args.config) + ".yaml", "r") as config_file:
            config = yaml.load(config_file, Loader=yaml.FullLoader)
    except Exception as e:
        traceback.print_exc()
        print(e)
        raise RuntimeError(
            "Config not found or unprovided, a configuration JSON path is REQUIRED to run"
        )
    from lightning_module import DIM_h5_Data_Module

    Dataset = DIM_h5_Data_Module(
        config,
        max_len=25,
        ignore_errors=True,
        overwrite=False,
    )
