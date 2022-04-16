# Mugnier: A general python package to simulate open quantum systems

A *restart* of `minimisTN`. Seperate graph-related algorithm and tensor operations with the help of GPU/NPU accelerated backend for better performace.


## Outlines & Style



- structure.network
- Method: 算法的抽象类


## Setup

- Development setup: 
    
    0. Create python virtural environment with python vesion >= 3.10.

    1. Prepare dependencies: `numpy`, `pytorch`.

        (For Apple silicon, complie NumPy locally to make use of `Accelerate` and AMX for best performace.)

    2. Install `mugnier` in develop mode using `pip`:

            pip install -e .

    3. For testing examples, install `jupyter-lab` and `matplotlib`.

