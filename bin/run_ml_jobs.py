#!/usr/bin/env python3

from collections import namedtuple
import glob
import os
import shutil

import common
from common import PATHS, BENCHMARKS_INFO, ML_INPUT_PARTIONS

Job = namedtuple('Job', ['benchmark', 'hard_brs_file', 'experiment_name', 'config_file', 'training_mode'])
JOBS = [
    Job('leela', 'top3', 'testrun2', 'mini_250', 'mini')
]

BATCH_SIZE = 2048
TRAINING_STEPS = [100, 100, 100]
FINE_TUNING_STEPS = [50, 50, 50]
LEARNING_RATE = 0.1
LASSO_COEFFICIENT = 0.0
REGULARIZATION_COEFFICIENT = 0.0
CUDA_DEVICE = 0
LOG_VALIDATION = False

CREATE_WORKDIRS = True
WORKDIRS_OVERRIDE_OK = True

def create_run_command(workdir, training_datasets, evaluation_datasets,
                       validation_datasets, br_pc, training_mode):
    return ('cd {workdir}; python run.py '
            '-trtr {tr} -evtr {ev} -vvtr {vv} --br_pc {pc} --batch_size {batch} '
            '-bsteps {bsteps} -fsteps {fsteps} --log_progress {log_validation} '
            '-lr {lr} -gcoeff {gcoeff} -rcoeff {rcoeff} -mode {mode} '
            '-c config.yaml --cuda_device {cuda} &> run_logs/{pc}.out'.format(
                workdir=workdir,
                tr=' '.join(training_datasets),
                ev=' '.join(evaluation_datasets),
                vv=' '.join(validation_datasets),
                pc=hex(br_pc),
                batch=BATCH_SIZE,
                bsteps=' '.join(map(str, TRAINING_STEPS)),
                fsteps=' '.join(map(str, FINE_TUNING_STEPS)),
                log_validation='--log_validation' if LOG_VALIDATION else '',
                lr=LEARNING_RATE,
                gcoeff=LASSO_COEFFICIENT,
                rcoeff=REGULARIZATION_COEFFICIENT,
                mode=training_mode,
                cuda=CUDA_DEVICE,
            ))

def get_workdir(job):
    return '{}/{}/{}'.format(PATHS['experiments_dir'], job.experiment_name, job.benchmark)

def create_workdirs():
    if not WORKDIRS_OVERRIDE_OK:
        for job in JOBS:
            workdir = get_workdir(job)
            assert not os.path.exists(workdir), 'Experiment already exists at {}'.format(workdir)

    for job in JOBS:
        workdir = get_workdir(job)
        branchnet_dir = os.path.dirname(os.path.abspath(__file__)) + '/../src/branchnet'
        os.makedirs(workdir, exist_ok=True)
        os.makedirs(workdir + '/run_logs', exist_ok=True)
        os.makedirs(workdir + '/visual_logs', exist_ok=True)
        for filename in ['run.py', 'dataset_loader.py', 'model.py']:
            shutil.copy(branchnet_dir + '/' + filename, workdir)
        shutil.copyfile('{}/configs/{}.yaml'.format(branchnet_dir, job.config_file), workdir + '/config.yaml')


def create_job_commands():
    cmds = []
    for job in JOBS:
        workdir = get_workdir(job)
        hard_brs = common.read_hard_brs(job.benchmark, job.hard_brs_file)
        datasets_dir = '{}/{}'.format(PATHS['ml_datasets_dir'], job.benchmark)
        training_datasets = [dataset
                             for inp in ML_INPUT_PARTIONS[job.benchmark]['training_set']
                             for dataset in glob.glob('{}/{}_{}_*.hdf5'.format(datasets_dir, job.benchmark, inp))]
        evaluation_datasets = [dataset
                               for inp in ML_INPUT_PARTIONS[job.benchmark]['test_set']
                               for dataset in glob.glob('{}/{}_{}_*.hdf5'.format(datasets_dir, job.benchmark, inp))]
        validation_datasets = [dataset
                               for inp in ML_INPUT_PARTIONS[job.benchmark]['validation_set']
                               for dataset in glob.glob('{}/{}_{}_*.hdf5'.format(datasets_dir, job.benchmark, inp))]

        for br in hard_brs:
            cmd = create_run_command(workdir, training_datasets, evaluation_datasets,
                                     validation_datasets, br, job.training_mode)
            cmds.append(cmd)
    return cmds


def main():
    if CREATE_WORKDIRS:
        create_workdirs()

    cmds = create_job_commands()
    common.run_parallel_commands_local(cmds, 1)


if __name__ == '__main__':
    main()