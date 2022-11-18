from pymatgen.transformations.standard_transformations import *
from pymatgen.transformations.advanced_transformations import *
import h5py
import numpy as np
import pickle as pk
import matminer.featurizers.structure as struc_feat
from multiprocessing import Process, Pool, cpu_count
from matbench.bench import MatbenchBenchmark
import scipy as sp
import argparse
import sys
from os.path import exists
from tqdm import tqdm
class OrthorhombicSupercellTransform(AbstractTransformation):
    """
    This transformation generates a combined scaled and shear of the provided unit cell to achieve a roughly
    OrthorhomicSupercell with roughly equal side lengths. Gets more accurate the larger the difference in size between primitive and supercell
    Robust alternative to the existing cubic supercell method that guarantees the inverse matrix is not singular when rounded. 
    """

    def __init__(self, N_Atoms):
        """
        Args:
            charge_balance_sp: The desired number of atoms in the supercell
        """
        self.N_Atoms = int(N_Atoms)

    def apply_transformation(self, structure):
        """
        Applies the transformation.

        Args:
            structure: Input Structure

        Returns:
            OrthorhombicSupercell
        """

        lattice_matrix = structure.lattice.as_dict()["matrix"]

        #RQ decomposition in this context provides a scale and shear matrix (R) that maps a Orthorhombic cell of unit volume to the current lattice parameters
        R, Q = sp.linalg.rq(lattice_matrix)
        #Invert R to get the scale+shear that maps the current unit cell to the Orthorhombic cell
        R1 = np.linalg.inv(R)
        #R1 is the inverse of R, we require the inverse of the diagonal component of R1 to remove the unwanted normalization included in the rq algorithm
        R1_Diagonal = np.zeros(R1.shape)
        np.fill_diagonal(R1_Diagonal,np.diagonal(R1))
        #S is the 'ideal' normalized shearing, it is not yet suitable due to its non-integer components
        S = sp.linalg.inv(R1_Diagonal) @ R1

        #The lattice parameters of Q are the "ideal" attained by directly applying S, we compute our scaling matrix by iteratively incrementing the shortest lattice parameter on Q
        #until any further increments breach the upper atom limit. These increments on Q are used to compute the scaling component of the transformation
        start_len = len(structure)
        Sheared_cell = S @ lattice_matrix
        Sheared_abc = [np.linalg.norm(Sheared_cell[0]),np.linalg.norm(Sheared_cell[1]),np.linalg.norm(Sheared_cell[2])]
        increments = (1,1,1)
        found_transform = False
        #Iteratively increment the shortest lattice parameters until doing so brings the number of atoms above the limit
        while not found_transform:
            new_increments = list(increments) #Deep copy
            shortest = np.argmin([i*j for i,j in zip(increments,Sheared_abc)]) #Return the shortest lattice parameter of Q post scaling
            new_increments[shortest] += 1
            if np.prod(new_increments)*start_len <= self.N_Atoms: #If this increment brings the total number of atoms above the ceiling then return the transformation matrix, otherwise repeat
                increments = new_increments
            else:
                found_transform=True
        cubic_upscale_approx = np.rint(np.diag(increments) @ S) # Create combined scale and shear matrix and round the off diagonals to the nearest integer, this provides an integer approximation of the shear, larger supercells will be more precise
        structure = SupercellTransformation(scaling_matrix=cubic_upscale_approx).apply_transformation(structure) # Apply the computed integer supercell transformation
        return structure

    @property
    def inverse(self):
        """Returns: None"""
        return None

    @property
    def is_one_to_many(self):
        """Returns: False"""
        return False

#Processes the pymatgen structure provided, first transforming to a primitive, and then upscaling to a cubic supercell of a given size if enabled
class process_structures():
    def __init__(self,cubic_supercell,supercell_size):
        self.cubic_supercell = cubic_supercell
        self.supercell_size = supercell_size
    def process_structure(self,struct):
        primitive_trans = PrimitiveCellTransformation()
        struct = primitive_trans.apply_transformation(struct)
        prim_size = len(struct)
        if self.cubic_supercell:
            orthog = OrthorhombicSupercellTransform(self.supercell_size)
            supercell = orthog.apply_transformation(struct)
        else:
            supercell = struct
        images = len(supercell)//len(struct)
        #Matbench band gap dataset does not contain disorder, but this code is general
        if not supercell.is_ordered:
            oxi_dec = AutoOxiStateDecorationTransformation()
            supercell= oxi_dec.apply_transformation(supercell)
            order_trans = OrderDisorderedStructureTransformation()
            supercell = order_trans.apply_transformation(supercell)
        return supercell,prim_size,images

def generate_crystal_dictionary(struc_and_target):
    struc = struc_and_target[0]
    composition = struc.composition.formula
    crystal_dict = {
        "structure": struc,
        "composition": composition,
        "target": struc_and_target[1],
        "prim_size": struc_and_target[2],
        "images": struc_and_target[3],
    }
    return crystal_dict


def divide_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i : i + n]

def h5_dataset_from_structure_list(hdf5_file_name, structure_dictionary,cpus):
    f = h5py.File(hdf5_file_name, "w", libver="latest")
    keys_list = list(structure_dictionary.keys())
    keys_chunked = list(divide_chunks(keys_list, 2048))
    for keys in tqdm(keys_chunked):
        values = [structure_dictionary[key] for key in keys]
        pool = Pool(processes=cpus)
        processed_values = [
            i
            for i in tqdm(
                pool.imap(
                    generate_crystal_dictionary,
                    values,
                )
            )
        ]
        pool.close()
        pool.join()
        pool.terminate()
        # crystal dict of dicts
        cdd = {key: value for key, value in zip(keys, processed_values)}
        print("dumping processed structures to hdf5 file")
        for key in tqdm(cdd.keys()):
            group = f.require_group(str(key))
            group.create_dataset("composition", data=cdd[key]["composition"])
            group.create_dataset(
                "pymatgen_structure", data=np.void(pk.dumps(cdd[key]["structure"]))
            )
            group.create_dataset("target", data=cdd[key]["target"])
            group.create_dataset("prim_size", data=cdd[key]["prim_size"])
            group.create_dataset("images", data=cdd[key]["images"])

def dataset_to_hdf5(inputs,outputs,h5_file_name,cpus,fold_n,supercell,supercell_size):
    #Create tuples of crystal index names, pymatgen structures, and properties
    structure_list = [(i, j, k) for i, j, k in zip(inputs.index, inputs, outputs)]
    #Transform the structures into primitive unit cells, and then upscale if appropiate
    processor = process_structures(supercell,supercell_size)
    pool = Pool(processes=cpu_count())
    processed_structures = [
        i for i in tqdm(pool.imap(processor.process_structure, [i[1] for i in structure_list]))
    ]
    pool.close()
    pool.join()
    pool.terminate()
    #Create tuple of processed structure, target proprety, size of primitive unit cell, and number of images
    processed_structures = [
        (processed_structures[i][0], structure_list[i][2],processed_structures[i][1],processed_structures[i][2])
        for i in range(len(structure_list))
    ]
    #Create a dictionary mapping each dataset index to the generated tuples
    structure_dict = {i[0]: j for i, j in tqdm(zip(structure_list, processed_structures))}
    #Initialize the h5 database with the pymatgen structures, the target, the primitive size, and the number of images
    if not exists("Data/Matbench/" + h5_file_name + "_" + str(fold_n) + ".hdf5"):
        h5_dataset_from_structure_list("Data/Matbench/" + h5_file_name + "_" + str(fold_n) + ".hdf5", structure_dict,cpus)

if __name__ == "__main__":
    #Arguments for whether to generate primitive cells or supercells, and what size the supercells should be capped at
    parser = argparse.ArgumentParser(description="ml options")
    parser.add_argument('--cubic_supercell', default=False, action='store_true')
    parser.add_argument('--primitive', default=False, action='store_true')
    parser.add_argument("-s", "--supercell_size", default=100,type=int)
    parser.add_argument("-w", "--number_of_worker_processes",default = 1,type=int)
    args = parser.parse_args()  
    if args.cubic_supercell:
        h5_file_name = "matbench_mp_gap_cubic_" + str(args.supercell_size)
        supercell = True
        supercell_size = args.supercell_size
    elif args.primitive:
        h5_file_name = "matbench_mp_gap_primitive"
        supercell = False
        supercell_size = None
    else:
        raise(Exception("Need to specify either --primitive or --cubic_supercell on commandline, with -s argument controlling supercell size"))
    task = MatbenchBenchmark().matbench_mp_gap
    task.load()
    fold_n = 1
    for fold in task.folds[:1]:
        #Get the data from matbench
        train_inputs, train_outputs = task.get_train_and_val_data(fold)
        test_inputs,test_outputs = task.get_test_data(fold,include_target=True)
        #Process the pymatgen structures and generate hdf5 database for training
        dataset_to_hdf5(train_inputs,train_outputs,h5_file_name + "_train",args.number_of_worker_processes,fold_n,supercell,supercell_size)
        dataset_to_hdf5(test_inputs,test_outputs,h5_file_name + "_test",args.number_of_worker_processes,fold_n,supercell,supercell_size)        
        fold_n += 1