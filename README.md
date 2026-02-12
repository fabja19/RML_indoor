This repository contains the code used for the paper "Radio Map Prediction from Noisy Environment Information and Sparse Observations" (soon on arXiv and under revision at IEEE). Since in reality the available information of the environment is often incomplete, inaccurate or partially outdated, we investigate whether CNNs can be trained with simulated noise as data augmentation (SNDA) in order to make them robust to similar noise at test time as well. The noise/perturbations we consider are shifted object and Tx locations and perturbed material properties.

# Overview of the Code
- *lib* contains model architectures, dataset and training/testing logic
- the scripts in the main folder are used to run training/testing with different parameters for trained models and baselines
- *data_seminar_room* contains the data from measurements and RT simulations
- the dataset from TBD is expected to be placed under *./dataset*
- *env* contains .yml files describing the mamba env we used
- *notebooks/nb_test_seminar_room.ipynb* contains the code used for test on real world measurements

# Links
- the code for the generation of indoor environments is available at https://github.com/fabja19/WI_indoor_projects
- the dataset will be uploaded in the next days to Zenodo, we will place the link here