# BranchNet
This repository contains the source code for the evaluation infrastructure of [BranchNet](https://www.microarch.org/micro53/papers/738300a118.pdf), published in 53rd IEEE/ACM International Symposium on Microarchitecture MICRO-53, 2020.

Citation: 
Siavash Zangeneh, Stephen Pruett, Sangkug Lym, and Yale N. Patt, “BranchNet: A Convolutional Neural Network to Predict Hard-To-Predict Branches,” in *the Proceedings 53rd Annual IEEE/ACM International Symposium on Microarchitecture*. MICRO-53 2020, Global Online Event, Oct 2020, pp. 118-130.

This repository contains most of the scripts that I used for producing the results in the paper. I did not include some scripts that were specific to my computing environment (e.g. a script for automatically running SPEC benchmarks) or I thought did not provide value for the community (e.g. wrappers around the provided scripts for sensitivity studies, running motivation experiments). But if you think something is missing, feel free to open an issue or email me, I'll either add them to the repo or at least guide you on what to do. Also, feel free to reach out to me with any questions about the paper.

## Dependencies 

* Linux (I mainly use CentOS)

* Python packages, with the versions that I've used:
```
Package         Version
--------------- -------
h5py            2.10.0
matplotlib      3.1.1
numpy           1.17.2
PyYAML          5.1.2
torch           1.3.0
torchvision     0.4.1
```

* Intel Pin Tool for generating branch traces (I've tested with 3.5 and 3.11)

* cmake3 and make

## Repository Overview

### src/branchnet

This directory contains the source code of the neural network model.

**dataset_loader.py**: reads datasets in a custom format from files and produces PyTorch Datasets.

**model.py**: defines a BranchNet PyTorch model.

**run.py**: the main script that connects everything together to do training and evaluation with checkpointing support.

**configs**: this subdirectory contains knobs for defining BranchNet models.

For running jobs, refer to the commandline documentation:
```
python run.py --help
```
You do not *have to* use this script directly. There's a wrapper script in the bin directory for launching multiple jobs.

### src/tracer

This directory is a pintool for creating branch traces. You could invoke *make* to build it, or use the helper scripts in the bin directory.

### src/tage

This directory contains the source code for runtime predictors: TAGE-SC-L and MTAGE-SC. You could use *cmake* to build them, or use the helper scripts in the bin directory.

### src/include

Common headers (defines branch trace format).

### environment_setup

This file contains the global paths that you need to define for using the helper scripts. I have committed example dummy version. You need to make a copy of each file, rename them to remove "_example", and provide your paths and benchmark definitions. See *How to run experiments* for more detailed.

### bin

This directory contains helper scripts for launching experiments. See *How to run experiments* for more details.

## How to run experiments?

### Create branch traces

First, build the tracer
```
./bin/build_tracer.py
```

In *./environment_setup/paths.yaml*, define *branch_trace_dir*: absolute path to a directory for storing branch traces for benchmarks.

Create your *./environment_setup/benchmarks.yaml*. Use the example for format. Right now, only simpoints with "type: pinball" are supported (i.e., the path in each simpoint points to a pinball). See below for other types.

Open *./bin/create_branch_traces.py* and edit TARGET_BENCHMARKS to include your benchmarks, and set NUM_THREADS according to your system capacity.

Run:
```
./bin/create_branch_traces.py
```

This will create traces in the *branch_traces_dir* directory, with the following structure. 
```
br_traces/
br_traces/benchmark1/benchmark1_(input_name)_simpoint(id)_brtrace.bz2
br_traces/benchmark2/benchmark2_(input_name)_simpoint(id)_brtrace.bz2
br_traces/benchmark3/benchmark3_(input_name)_simpoint(id)_brtrace.bz2
```

If you cannot use this script, but want to use the rest of the scripts, that is fine as long as you follow the same naming conventions when you create the branch traces yourself. However, I suggest you to add a new type in *benchmarks.yaml* and modify *./bin/create_branch_traces.py* to run the appropriate commands for you. For example, you may want to attach the pintool to a process and fast-forward to your region.


### Evaluate runtime predictors on the traces

First, build the runtime predictors
```
./bin/build_tage.py
```

In *./environment_setup/paths.yaml*, define *tage_stats_dir*: absolute path to a directory for storing the results of runtime predictors for each benchmark.

Open *./bin/run_tage.py* and edit TARGET_BENCHMARKS to include your benchmarks of interest, BINARY_NAME to any of the binary names produces by *build_tage.py*, CONFIG_NAME to any name you like (this will be used by other script to refer to the runtime predictor results), and set NUM_THREADS according to your system capacity.

Run:
```
./bin/run_tage.py
```

### Identify hard-to-predict branches 

Create your *./environment_setup/ml_input_partitions.yaml*. Use the example for format. Note that ML benchmark names could optionally be different from the main benchmark names, thus, there is a field named *spec_name* that points each ML benchmark to its corresponding main benchmark name. This is useful if you want to run experiment where you use treat a program with a different reference input as an independent benchmark for ML training.

In *./environment_setup/paths.yaml*, define *hard_brs_dir*: absolute path to a directory for storing the PCs of hard-to-predict branches for each benchmark.

Open *./bin/print_hard_brs.py* and edit TARGET_BENCHMARKS to include your ML benchmarks of interest, TAGE_CONFIG_NAME to the CONFIG_NAME used for producing the runtime predictor results, NUM_BRS_TO_PRINT to number of branches, PRODUCE_HARD_BR_FILES to True (otherwise the script simply prints branch statistics), and HARD_BR_FILE_NAME to a name for your branch selection.

Run:
```
./bin/print_hard_brs.py
```

Note that the hard_br files are simple text files, where each line is a branch PC in hexadecimal. You can modify these manually, too, without running the script, which is useful when targetting specific branches for short experiments.

### Create BranchNet datasets

We need to convert the branch traces to a format that is more suitable for training. The key idea is to find the occurances of the hard-to-predict branches and store their positions in the trace along with the trace. We store the traces in *hdf5* file format.

In *./environment_setup/paths.yaml*, define *ml_datasets_dir*: absolute path to a directory for storing the branch traces, ready for running ML jobs in *hdf5* file format.

Open *./bin/create_ml_datasets.py* and edit TARGET_BENCHMARKS to include your ML benchmarks of interest, HARD_BR_FILE_NAME to the name you used for your hard-to-predict branch selection, PC_BITS to the number of least significant bits in the PC to keep, and set NUM_THREADS according to your system capacity.

Run:
```
./bin/create_ml_datasets.py
```

### Run BranchNet training and evaluation jobs

Finally we're ready to actually training and evaluate a neural network model!

In *./environment_setup/paths.yaml*, define *experiments_dir*: absolute path to a directory for work directories and results of the ML jobs.

Open *./bin/run_ml_jobs.py* and edit JOBS to define your experiments. JOBS is a list of *Job* namedtuples. The definition of the fields
* benchmark: name of an ML benchmark (as defined in *./environment_setup/ml_input_partitions.yaml*)
* hard_brs_file: name associated with the hard_br files (as chosen in *Identify hard-to-predict branches*)
* experiment_name: a unique new name for your experiment. 
* config_file: name of a config file in *./src/branchnet/configs/* without .yaml suffix. 
* training_mode: Affects how the neural network is quantized. Choose between {big, mini, tarsa}. *big* is a good default as it does not do any quantization. See *./src/branchnet/run.py* for its effect.

Run:
```
./bin/run_ml_jobs.py
```

### Printing MPKI Reduction
Open *./bin/print_results.py* and edit SUITE, CONFIGS, BUDGET, TAGE_CONFIG_NAME, CSV, DUMP_PER_BR_STATS
* SUITE: all benchmarks that are evaluated together.
* CONFIGS: list of available BranchNet models and their required budget.
* BUDGET: Total budget. 
* TAGE_CONFIG_NAME: name of the config used for producing the runtime predictor results.
* CSV: Controls the output format. Could be True or False.
* DUMP_PER_BR_STATS: write per-branch statistics of each benchmark in a file.

The reason SUITE and CONFIGS are a list is to support heterogenous Mini-BranchNet evaluation. For most typical use cases, simply have one item in CONFIGS for your single model, assign it a budget of 1, and set BUDGET to the maximum number of models that you want to support.

Run:
```
./bin/print_results.py
```
