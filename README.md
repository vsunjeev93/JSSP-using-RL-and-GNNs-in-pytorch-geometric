# Job shop scheduling using RL and GNNs
A  GNN based RL algorithm for job shop scheduling leveraging the pytorch geometric framework.\


# Reinforcement Learning for Job Shop Scheduling Problem

This repository implements a reinforcement learning solution for the Job Shop Scheduling Problem (JSSP) using Graph Neural Networks (GNNs). The implementation uses PyTorch and PyTorch Geometric for graph-based deep learning.

## Problem Description

The Job Shop Scheduling Problem (JSSP) is a classical optimization problem where we need to schedule a set of jobs on a set of machines. Each job consists of a sequence of operations that must be processed in a specific order, and each operation must be processed on a specific machine for a given duration. The goal is to minimize the makespan (total completion time) while satisfying all constraints.

## Architecture

The solution uses an Actor-Critic reinforcement learning architecture with Graph Neural Networks

## Requirements
- PyTorch
- PyTorch Geometric
- NumPy
- CUDA (optional)

### Graph Representation
Each JSSP instance is represented as a graph (see figure below) where:
![image](https://github.com/user-attachments/assets/f8f68257-dabe-47e9-870a-832767b62105)

- **Nodes:** Operations of jobs, including special start (0) and end (n+1) nodes

- **Edges:** Three types of connections in the graph:
  1. **Precedence Edges:** Fixed connections between operations in the same job
  2. **Machine Edges:** Dynamic connections between operations on the same machine
  3. **Reverse Edges:** Mirrored edges for bidirectional message passing in the GNN

- **Node features:** Vector of length (nm + 3) containing:
  - Processing time
  - Remaining processing time after this operation
  - Steps remaining in job
  - Machine assignment (one-hot encoded)


### Components

1. **Actor Network** (`actor.py`):
   - Uses two Graph Isomorphism Networks (GIN) for bidirectional message passing
   - Combines node embeddings with graph-level embeddings
   - Outputs action probabilities for selecting the next operation

2. **Critic Network** (`critic.py`):
   - Single GIN network for state evaluation
   - Estimates value function for the current state

3. **GIN Module** (`GIN.py`):
   - Implementation of Graph Isomorphism Network
   - Uses MLPs with batch normalization and ReLU activations
   - Provides both node-level and graph-level embeddings

4. **State Transition** (`state_transition.py`):
   - Handles the environment dynamics
   - Updates machine availability times
   - Manages operation dependencies and constraints

5. **Data Generator** (`data_generator.py`):
   - Generates random JSSP instances
   - Creates graph representations of the problems
   - Supports batch processing for training

## Usage

### Training
```bash
python train_JSSP.py --nj 10 --nm 10 --instances 300 --batch_size 10
```

### Testing
```bash
python test.py --nj 10 --nm 10 --instances 100 --batch_size 1
```
#### test results
6x6- 574.14
10x10- 1059.79

### Parameters
- `nj`: Number of jobs
- `nm`: Number of machines
- `instances`: Training instances per epoch
- `batch_size`: Batch size
- `seed`: Random seed
- `low`: Lower bound for processing times
- `high`: Upper bound for processing times



