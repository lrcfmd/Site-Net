Implementation of "Site-Net: Using global self-attention and real-space supercells to capture long-range interactions in crystal structures" (https://arxiv.org/abs/2209.08190). Implemented in pytorch lightning using hdf5 as the data storage solution. Performance improvements that remove redundant calculations have been added to make training significantly faster. These optimizations will be exapnded on in the full publication.

## Environments

A conda environment .yaml file has been included which will allow the construction of an anaconda environment. This conda environment can be created using

##### conda env create -f sitenet_env.yaml

For this file to resolve properly the channel priority on conda must be set to strict. For convinience, create_sitenet_env.sh has been provided. This .sh file will store the current value of the channel priority, set the channel priority to strict, and then revert it back to whatever the previous setting was after the environment is installed. By default the name of the virtual environment will be sitenet_env.yaml and can be activated with 

##### conda activate sitenet_env

This local environment contains the same versions of all key packages used to generate the original results

## Scripts for reproducing paper results

### create_mp_gap_hdf5.py

Produce an hdf5 file for reproducing paper results

##### \--primitive generates a dataset of primitive unit cells

##### \--cubic_supercell generates a dataset of supercells

##### -s \--supercell_size allows the size of the supercells to be specified 

##### -w \--number_of_worker_processes allows the number of cpu threads used to be specified (default 1)

either \--primitive or \--cubic_supercell must be used

provide the size of the supercell (if applicable) with -s N where N is the maximum number of atoms

### train.py

This script trains a site-net model on the provided hdf5 dataset using the hyper parameters defined in the config yaml

##### -c \--config allows the path of the configuration file to be specified (default None) 

##### -f \--h5_file_name allows the path of the h5 dataset used for training to be specified (default None) 

##### -l \--load_checkpoints allows training to be resumed from the most recent checkpoint for a given config file (default 0) 

##### -o \--overwrite will force the generation of new features, followed by overwriting, instead of reading them from the h5 file (default False) 

##### -n \--limit will limit the model to loading the first n samples, this is normally for debug purposes / test runs (default None) 

##### -u \--unit_cell_limit will exclude unit cells larger than this size from training (default 100) 

##### -w \--number_of_worker_processes controls the maximum number of cpu threads that site-net will use (default 1)

### predict_reproduce.py

Runs inference on the data provided using the config and model parameters provided. The MAE will be printed, and the data neccessary to reproduce the plots in the paper will be saved. This file also contains some ad hoc adjustments to the model so that it logs the attention coefficients and returns them

##### -c \--config allows the path of the configuration file to be specified (default None) 

##### -f \--dataset allows the path of the h5 dataset used for training to be specified (default None) 

##### -n \--limit allows a smaller subset of the data to be used for the inference (default None) 

##### -m \--model_name allows the checkpoint path to be specified (default None) 

##### -w \--number_of_worker_processes allows the number of cpu threads to be specified (default 1)

### plot_reproduce.py

After running predict_reproduct.py this script will output the plots to the histograms folder

### Training a band gap model and reproducing plots

Steps for benchmarking a 100 atom supercell model on the first fold of the band gap task, 100 is less intensive to run than 500.

This implementation of Site-Net uses the hdf5 format for storing datasets, to initialize the hdf5 file first run

##### python create_mp_gap_hdf5.py --cubic_supercell -s 100

This will create files "Data/Matbench/matbench_mp_gap_cubic_100_train_1.hdf5" and "Data/Matbench/matbench_mp_gap_cubic_100_test_1.hdf5". These will contain the structure objects but will not contain any features.

The model can be trained with

##### python train.py -c config/PaperParams.yaml -f Data/Matbench/matbench_mp_gap_cubic_100_train_1.hdf5 -u 100 -w [number of cpu threads available]

This will train a model using the hyper parameters from the paper (Table 1), on the training fold of the band gap task, limiting training to supercells of size 100 to ensure that all samples are coming from the same distribution. Featurization is performed "just in time" and results from featurizers are cached in the hdf5 file for later use, the first loading in of data by train.py will take a long time as it generates the features

Training can be tracked using tensorboard, the outputs are generated in the lightning_logs folder, which is where tensorboard should be pointed to. Avg_val_loss is the validation MAE on the task.

Model checkpoints are saved where the path is the "label" parameter in the config appended to the dataset file_name, the most recent checkpoint will be saved alongside the best validation score achieved, the checkpoint associated with the best achieved validation score will have "best" included in the checkpoint name. The models reported in the paper were the best validation scores achieved, and were stopped after the models had converged and the "best" checkpoint was stable

Once the model has trained the MAEs on the test fold can be obtained with.

##### python predict_reproduce.py -c config/PaperParams.yaml -m Data/Matbench/matbench_mp_gap_cubic_100_train_1.hdf5_best_PaperParams.ckpt -f Data/Matbench/matbench_mp_gap_cubic_100_test_1.hdf5 -w [number of cpu threads available]

In Data/checkpoints the reported model checkpoints can be found, the prefix is the training method (N atom supercell vs geometric cutoff) and the suffix is the ammount of data used (full is 10^5, medium is 10^4 and small is 10^3). Inference can be run on all of these checkpoints with the correct config file and dataset to reproduce table 2 and 3.

Running predict.py also dumps stats about the model that were used to construct paper figures such as the 2d histograms from the paper that show attention coefficients versus distance between atoms. These plots can be generated from the most recent run of predict.py by running plots.py

plots.py accepts no arguments as it will read the most recent outputs of predict.py. The outputs of plots.py will be generated in the histograms folder. The file TrueVPred.png is figure 5. The file distance_coefficients.png is figure 6.

The lightning module is in lightning_module.py, individual torch modules are in modules.py, h5_handler.py contains the database management code, none of these files are to be run directly.

### Using your own data

#### cif_zip_to_hdf5.py
 
Produce a hdf5 file ready for use with train.py and predict.py using a zip of cif files and a csv defining supervised properties. Does not currently support disordered structures or multiple objectives. This script isn't used in the paper, rather, it exists as a way to run site-net on arbitrary datasets by mapping the properties and cif files to an hdf5 database. The zip file should only contain cif files and the csv should consist of a "file" column identifying the cif and "target" column specifying the value of the supervised property. An example setup with 2 cifs that have been assigned random target variables has been included in the repo.

\--primitive generates a dataset of primitive unit cells --cubic_supercell generates a dataset of supercells 

-s \--supercell_size allows the size of the supercells to be specified 

-w \--number_of_worker_processes allows the number of cpu threads used to be specified (default 1) 

-c \--cif_zip Provide path to cif zip file, this should be a zip file containings only .cif files

-d \--data_csv Provide path to csv containing a column called "file" containing cif file names and a column called "target" containing the associated supervised value 

-hd \--h5_path Provide path for the new hdf5 file

### predict_reproduce.py

Runs inference on the given hdf5 dataset using the model provided. Unlike predict_reproduce this does not generate any additional side effects. If you provide the csv file used as part of creating the dataset cif_zip_to_hdf5.py then the predicted values will be added as a new column to that file. This will allow general usage of the model.

-c \--config allows the path of the configuration file to be specified (default None) 

-f \--dataset allows the path of the h5 dataset used for training to be specified (default None) 

-n \--limit allows a smaller subset of the data to be used for the inference (default None) 

-m \--model_name allows the checkpoint path to be specified (default None) 

-w \--number_of_worker_processes allows the number of cpu threads to be specified (default 1)

## configuring the model
yaml is used to configure the models, the paper parameters are defined in the config folder and can be adjusted.

A User friendly way to change the featurizers used from the ones in the paper is a WIP. If you want to use your own featurizers, please check h5_handler.py where featurizers functions are in a dictionary generated by unpacking the modules directly. The dictionary is populated currently with matminer, dscribe and pymatgen featurizers.
