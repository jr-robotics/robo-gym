# Getting started using RLlib with robo-gym

## Prerequisites

Tested on Ubuntu 20.04 using Python 3.8.

### Installation of robo-gym

Install an up-to-date version of robo-gym-server-modules. Recommended to check out version 0.3.0.0 or newer and install from source using

    pip install -e .

Install robo-gym from the copy at hand using

    pip install -e .

### Installation of RLlib

Follow [instructions from RLlib](https://docs.ray.io/en/latest/rllib/index.html).

It is recommended to also install gputil for use by RLlib:

    pip install gputil

At runtime, you may encounter this error:

    (...) ray/_private/accelerators/nvidia_gpu.py", line 71, in get_current_node_accelerator_type
        device_name = device_name.decode("utf-8")
    UnicodeDecodeError: 'utf-8' codec can't decode byte 0xf8 in position 0: invalid start byte

It is typically resolved by updating the NVidia GPU driver. If occuring in an WSL environment, this typically concerns the driver for Windows.

## Using RLlib with robo-gym

The scripts here are based on examples from RLlib. They are tested against Ray/RLlib v2.1.0. The API may deviate if you use a different version.

Mind that setting the environment argument `rs_state_to_info` in the initialization of UR environments to a value of `False` is important for being able to save checkpoints.

### Scripts for training followed by inference

* [Training, saving and inference for MiR100](./training_inference_mir.py)
* [Training, saving and inference for UR5](./training_inference_ur.py)

Each training script does a few inference runs in its last phase, applying the trained model to a new instance of the respective environment. The training scripts support a number of arguments to customize the training and inference runs. Call the script with the argument `-h` or `--help` to see the available arguments.

In addition, the environment variable `RLLIB_NUM_GPUS` can be set to the desired value (default: 0) to set the number of GPUs that RLlib should use.

The inference scripts for the policy-only approach may need to be adjusted if a non-default framework (i.e., not Torch) was used for the training.

Note that the default environment registration from Gymnasium (establishing the link from environment names to the implementing Python classes) is not sufficient for using the environments in RLlib agents. This is why each of the training scripts uses RLlib's own registration mechanism and consequently has to import the respective environment class. 

### Scripts for inference from policy checkpoint

The saved checkpoints from the training scripts can also be used in the standalone inference scripts.

Pass the policy checkpoint path as an argument to the script. If using a policy from an algorithm checkpoint, a typical folder for a policy checkpoint is obtained by appending policies/default_policy to the path.

* [Inference from a saved policy checkpoint for MiR100](./policy_from_cp_mir.py)
* [Inference from a saved policy checkpoint for UR5](./policy_from_cp_ur.py)

Inference from saved algorithm checkpoints is possible as an alternative to the shown restoring of policy checkpoints. The inference code can be derived from the inference section of the training scripts. However, configured algorithms in RLlib contain their environment. For robo-gym environments, this implies that a simulation server is started for each worker instance.