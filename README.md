# Asynch_mb
The code is written in Python 3 and builds on [Tensorflow](https://www.tensorflow.org/). 
Many of the provided reinforcement learning environments require the [Mujoco](http://www.mujoco.org/) physics engine.
Overall the code was developed under consideration of modularity and computational efficiency.
Many components of the Meta-RL algorithm are parallelized either using either [MPI](https://mpi4py.readthedocs.io/en/stable/) 
or [Tensorflow](https://www.tensorflow.org/) in order to ensure efficient use of all CPU cores.

## Installation / Dependencies
The provided code can be either run in A) docker container provided by us or B) using python on
your local machine. The latter requires multiple installation steps in order to setup dependencies.

##### Installing MPI
Ensure that you have a working MPI implementation ([see here](https://mpi4py.readthedocs.io/en/stable/install.html) for more instructions). 

For Ubuntu you can install MPI through the package manager:

```
sudo apt-get install libopenmpi-dev
```

###### Anaconda 
If not done yet, install [anaconda](https://www.anaconda.com/) by following the instructions [here](https://www.anaconda.com/download/#linux).
Then reate a anaconda environment, activate it and install the requirements in [`requirements.txt`](requirements.txt).
```
conda env create -f environment.yml
```

##### B.4. Set up the Mujoco physics engine and mujoco-py
For running the majority of the provided Meta-RL environments, the Mujoco physics engine as well as a 
corresponding python wrapper are required.
For setting up [Mujoco](http://www.mujoco.org/) and [mujoco-py](https://github.com/openai/mujoco-py), 
please follow the instructions [here](https://github.com/openai/mujoco-py).
