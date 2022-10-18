import h5py
import numpy as np
import pickle as pk
from tqdm import tqdm
import matminer.featurizers.structure as struc_feat
from multiprocessing import Process, Pool, cpu_count
from pymatgen.transformations.standard_transformations import *
from matbench.bench import MatbenchBenchmark
from pymatgen.transformations.advanced_transformations import *
import scipy as sp

class OrthorhombicSupercellTransform(AbstractTransformation):
    """
    This transformation generates a combined scaled and shear of the provided unit cell to achieve a roughly
    OrthorhomicSupercell. 
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
        #Compute the inverse of R1's diagonal and apply as a matrix multplication to remove scaling component from R1, as this is undesirable
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

def order_disordered(struct):
    #Unused here because none of the structures in the dataset are disordered
    primitive_trans = PrimitiveCellTransformation()
    struct = primitive_trans.apply_transformation(struct)
    prim_size = len(struct)
    orthog = OrthorhombicSupercellTransform(100)
    supercell = orthog.apply_transformation(struct)
    if not supercell.is_ordered:
        oxi_dec = AutoOxiStateDecorationTransformation()
        supercell= oxi_dec.apply_transformation(supercell)
        order_trans = OrderDisorderedStructureTransformation()
        supercell = order_trans.apply_transformation(supercell)
    images = len(supercell)//len(struct)
    return supercell,prim_size,images



def generate_dist_and_coulomb(struc_and_target):
    struc = struc_and_target[0]
    coulomb_matrix = struc_feat.CoulombMatrix(flatten=False,).featurize(
        struc
    )[0]
    distance_matrix = struc.distance_matrix
    composition = struc.composition.formula
    crystal_dict = {
        "structure": struc,
        "coulomb": coulomb_matrix,
        "distance": distance_matrix,
        "composition": composition,
        "target": struc_and_target[1],
        "prim_size": struc_and_target[2],
        "images": struc_and_target[3],
    }
    return crystal_dict


def divide_chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i : i + n]


def h5_dataset_from_structure_list(hdf5_file_name, structure_dictionary):
    f = h5py.File(hdf5_file_name, "w", libver="latest")
    print("Generating distance and coulomb matricies from structures")
    keys_list = list(structure_dictionary.keys())
    keys_chunked = list(divide_chunks(keys_list, 2048))
    for keys in tqdm(keys_chunked):
        values = [structure_dictionary[key] for key in keys]
        pool = Pool(processes=cpu_count())
        processed_values = [
            i
            for i in tqdm(
                pool.imap(
                    generate_dist_and_coulomb,
                    values,
                )
            )
        ]
        pool.close()
        pool.join()
        pool.terminate()
        # crystal dict of dicts
        cdd = {key: value for key, value in zip(keys, processed_values)}
        print("dumping structures + matricies to hdf5 file")
        for key in tqdm(cdd.keys()):
            group = f.require_group(str(key))
            # group.create_dataset("coulomb_matrix", data=cdd[key]["coulomb"])
            # group.create_dataset("distance_matrix", data=cdd[key]["distance"])
            group.create_dataset("composition", data=cdd[key]["composition"])
            group.create_dataset(
                "pymatgen_structure", data=np.void(pk.dumps(cdd[key]["structure"]))
            )
            group.create_dataset("target", data=cdd[key]["target"])
            group.create_dataset("prim_size", data=cdd[key]["prim_size"])
            group.create_dataset("images", data=cdd[key]["images"])

########################################################################################################################
from os.path import exists
if __name__ == "__main__":
    import pickle as pk
    from matbench.bench import MatbenchBenchmark
    import sys
    pool = Pool(cpu_count()//3)
    task = MatbenchBenchmark().matbench_mp_gap
    task.load()
    fold_n = 1
    for fold in task.folds[:1]:

        train_inputs, train_outputs = task.get_train_and_val_data(fold)

        test_inputs,test_outputs = task.get_test_data(fold,include_target=True)
        
        structure_list = [[i, j, k] for i, j, k in zip(train_inputs.index, train_inputs, train_outputs)]
        
        ordered_structures = [
            i for i in tqdm(pool.imap(order_disordered, [i[1] for i in structure_list]))
        ]
        print(ordered_structures[0])
        print(len(ordered_structures[0]))
        ordered_structures = [
            [ordered_structures[i][0], structure_list[i][2],ordered_structures[i][1],ordered_structures[i][2]]
            for i in range(len(structure_list))
        ]
        structure_dict = {i[0]: j for i, j in tqdm(zip(structure_list, ordered_structures))}
        if not exists("Data/CV_hdf5/matbench_mp_gap_cubic_COOs_train" + "_" + str(fold_n) + ".hdf5"):
            h5_dataset_from_structure_list("Data/CV_hdf5/matbench_mp_gap_cubic_COOs_train" + "_" + str(fold_n) + ".hdf5", structure_dict)

        structure_list = [[i, j, k] for i, j,k in zip(test_inputs.index, test_inputs,test_outputs)]
        
        ordered_structures = [
            i for i in tqdm(pool.imap(order_disordered, [i[1] for i in structure_list]))
        ]
        ordered_structures = [
            [ordered_structures[i][0], structure_list[i][2],ordered_structures[i][1],ordered_structures[i][2]]
            for i in range(len(structure_list))
        ]

        structure_dict = {i[0]: j for i, j in tqdm(zip(structure_list, ordered_structures))}
        if not exists("Data/CV_hdf5/matbench_mp_gap_cubic_COOs_test" + "_" + str(fold_n) + ".hdf5"):
            h5_dataset_from_structure_list("Data/CV_hdf5/matbench_mp_gap_cubic_COOs_test" + "_" + str(fold_n) + ".hdf5", structure_dict)
        
        fold_n += 1
    pool.close()
    pool.join()
    pool.terminate()