Implementation of "Site-Net: Using global self-attention and real-space supercells to capture long-range interactions in crystal structures" (https://arxiv.org/abs/2209.08190). Implemented in pytorch lightning using hdf5 as the data storage solution. Performance improvements that remove redundant calculations have been added to make training significantly faster. These optimizations will be exapnded on in the full publication.

#### Requirements ####

In addition to an anaconda environment on python 3.9.13 with standard packages

pytorch (With cuda)
pytorch lightning
torch scatter
dscribe
pymatgen
matminer
matbench
h5py
compress-pickle
#### Scripts and Arguments ####

=== cif_zip_to_hdf5.py ===

Produce a hdf5 file ready for use with train.py and predict.py using a zip of cif files and a csv defining supervised properties. Does not currently support disordered structures or multiple objectives.

--primitive generates a dataset of primitive unit cells
--cubic_supercell generates a dataset of supercells
-s --supercell_size allows the size of the supercells to be specified
-w --number_of_worker_processes allows the number of cpu threads used to be specified (default 1)
-c --cif_zip Provide path to cif zip file
-d --data_csv Provide path to csv containing a column called "file" containing cif file names and a column called "target" containin the associated supervised value
-hd --h5_path Provide path for the new hdf5 file

=== create_mp_gap_hdf5.py ===

Produce an hdf5 file for reproducing paper results

--primitive generates a dataset of primitive unit cells
--cubic_supercell generates a dataset of supercells
-s --supercell_size allows the size of the supercells to be specified
-w --number_of_worker_processes allows the number of cpu threads used to be specified (default 1)

either --primitive or --cubic_supercell must be used

provide the size of the supercell (if applicable) with -s N where N is the maximum number of atoms

=== train.py ===

-c --config allows the path of the configuration file to be specified (default None)
-f --h5_file_name allows the path of the h5 dataset used for training to be specified (default None)
-l --load_checkpoints allows training to be resumed from the most recent checkpoint for a given config file (default 0)
-o --overwrite will force the generation of new features, followed by overwriting, instead of reading them from the h5 file (default False)
-d --debug will limit the model to loading the first 1000 samples (default False)
-u --unit_cell_limit will exclude unit cells larger than this size from training (default 100)
-w --number_of_worker_processes controls the maximum number of cpu threads that site-net will use (default 1)

=== predict.py ===

-c --config allows the path of the configuration file to be specified (default None)
-f --dataset allows the path of the h5 dataset used for training to be specified (default None)
-n --limit allows a smaller subset of the data to be used for the inference (default None)
-m --model_name allows the checkpoint path to be specified (default None)
-w --number_of_worker_processes allows the number of cpu threads to be specified (default 1)
#### Steps for reproducing paper results, training + inference ####

Steps for benchmarking a 100 atom supercell model on the first fold of the band gap task, 100 is less intensive to run than 500

This implementation of Site-Net uses the hdf5 format for storing datasets, to initialize the hdf5 file first run

python create_mp_gap_hdf5.py --cubic_supercell -s 100

to create "Data/Matbench/matbench_mp_gap_cubic_100_train_1.hdf5" and "Data/Matbench/matbench_mp_gap_cubic_100_test_1.hdf5". These will contain the structure objects but will not be featurized.

Once this has been generated, the model can be trained with

python train.py -c config/PaperParams.yaml -f Data/Matbench/matbench_mp_gap_cubic_100_train_1.hdf5 -u 100 -w [number of cpu threads available]

featurization is performed "just in time" and results from featurizers are cached in the hdf5 file for later use, the first loading in of data by train.py will take a long time as it generates the features

Training can be tracked using tensorboard, the outputs are generated in the lightning_logs folder, which is where tensorboard should be pointed to

Model checkpoints are saved where the path is the "label" parameter in the config appended to the dataset file_name, the most recent checkpoint will be saved alongside the best validation score achieved, which will include "_best_". The models reported in the paper were the best validation scores achieved after the model had converged.

Once the model has trained the test score can be obtained with

python predict.py -c config/PaperParams.yaml -m matbench_mp_gap_cubic_100_train_1.hdf5_best_PaperParams.ckpt -f Data/Matbench/matbench_mp_gap_cubic_100_test_1.hdf5 -w [number of cpu threads available]

The attention coefficient plots and the performance plots can be obtained after running predict.py by running

python plots.py

the outputs will be generated in the histograms folder

The lightning module is in lightning_module.py, individual torch modules are in modules.py, h5_handler.py contains the database management code, none of these files are to be run directly.

#### configuration files ####

yaml is used to configure the models, the paper parameters are defined in the config folder and can be adjusted. 

User friendly way to change the featurizers used from the ones in the paper is a WIP. If you want to use your own featurizers, please check h5_handler.py where featurizers functions are in a dictionary that can be extended. The dictionary is populated currently with matminer, dscribe and pymatgen featurizers. 
