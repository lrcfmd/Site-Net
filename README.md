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
