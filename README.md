ICASSP2017 paper: DESIGNING EFFICIENT ARCHITECTURES FOR MODELING TEMPORAL FEATURES WITH CONVOLUTIONAL NEURAL NETWORKS
-----------------------------
This code can be easily adapted for music (spectrograms) classification using any deep learning architecture available in Lasagne-Theano.
It is build using python on Lasagne-Theano for deep learning and Essentia for feature extraction.

**Installation**
 
Requires having Lasagne-Theano (http://lasagne.readthedocs.org/en/latest/user/installation.html) and Essentia (http://essentia.upf.edu/documentation/installing.html) installed.

Lasagne is already in a folder that you can download together with MIRdl, to install Theano do: 
> sudo pip install --upgrade https://github.com/Theano/Theano/archive/master.zip

Dependencies: numpy and scipy.

**Important folders**
- *./data/datasets*: the library expects to have the dataset divided by folders that represent the tag to be predicted. 
- *./data/preloaded*: this directory contains the pickle files storing the datasets and confusion matrices in a readable format for the library. The name of the pickle file contains all the parameters used for computing it.
- *./data/results*: this directory stores the following files: **.result** (with training and test results), **.training** (having the training evolution, readable with utils.py!), **.param** (storing all the deep learning parameters used for each concrete experiment) and the **.npz** (where the best trained deep learning model is stored). The results and confusion matrices computed after cross-validation are also in that folder.
 
**Important scripts**
- *run.py*: where the network architecture is selected, you can also set the input and training parameters.
- *buildArchitecture.py*: where the Lasagne-Theano network architecture is set.
- *loadDatasets.py*: where audios are loaded, formatted and normalized to be fed into the net. 
- *dl.py*: main part of the library where the training happens.

**Reproducing the paper**
- run: *THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python run.py*. There, you can simply set the parameters and choose the architecture you want to use according to the paper. You will be able to reproduce all the results provided in the paper. The Ballroom dataset is also uploaded to this GitHub repository, after downloading it and installing the dependencies the experiments are ready to run.

**Steps for using this code (in general)**
- **0.0)** Install.

- **0.1)** Understand this tutorial: http://lasagne.readthedocs.org/en/latest/user/tutorial.html. This library is based on it!

- **1)** Download a dataset and copy it in *./data/datasets* (that repository already includes the Ballroom dataset). The library expects to have the dataset divided by folders that represent the tag to be predicted. 
For example, for the GTZAN dataset (http://marsyasweb.appspot.com/download/data_sets/) the library expects:
>./data/datasets/GTZAN/blues
>
>./data/datasets/GTZAN/classical
>
> (...)
>
>./data/datasets/GTZAN/rock
- **2)** Adapt the *load_datasets.py* function to work using your dataset.

- **3)** Set the *run.py* parameters and the deep learning architecture in *buildArchitecture.py*.

- **4)** Run *run.py*.

**License**

This code is Copyright 2016 - Music Technology Group, Universitat Pompeu Fabra. It is released under Affero GPLv3 license except for the third party libraries and datasets which have its own licenses.

This code is free software: you can redistribute it and/or modify it under the terms of the Affero GPLv3 as published by the Free Software Foundation. This code is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
