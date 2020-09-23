#!/usr/bin/env python3
'''
This script converts a sequential binary trace of branches produced by the 
tracer PIN tool to a HDF5 dataset and marks the hard to predict branches in
the trace. This format is more suitable for random accesses to arbitrary
instances of hard-to-predict branches during training.
'''

import bz2
import h5py
import multiprocessing
import numpy as np
import os
import struct

import common
from common import PATHS, BENCHMARKS_INFO


TARGET_BENCHMARKS = ['leela']
HARD_BRS_FILE = 'top100'
NUM_THREADS = 32
PC_BITS = 30


def read_branch_trace(trace_path):
    struct_type = [
        ('br_pc', np.uint64),
        ('target', np.uint64),
        ('dir', np.uint8),
        ('type', np.uint8)]
    record_dtype = np.dtype(struct_type, align=False)

    with bz2.open(trace_path, 'rb') as f:
        buffer = f.read()
    x = np.frombuffer(buffer, dtype=record_dtype)

    return x['br_pc'].copy(), x['dir'].copy()


def create_new_dataset(dataset_path, pcs, directions):
    '''
    Create a new hdf5 file and copy over the history to the file.
    Branch PCs and directions are concatenated. Only the least significant bits of PC
    (controlled by PC_BITS) are stored.
    '''
    stew_bits = PC_BITS + 1
    if stew_bits < 8:
        stew_dtype = np.uint8
    elif stew_bits < 16:
        stew_dtype = np.uint16
    elif stew_bits < 32:
        stew_dtype = np.uint32
    elif stew_bits < 64:
        stew_dtype = np.uint64
    else:
        assert False, 'History elements of larger than 64 bits are not supported'

    pc_mask = (1 << PC_BITS) - 1
    fptr = h5py.File(dataset_path, 'w-')
    processed_history = ((pcs & pc_mask) << 1) | directions
    processed_history = processed_history.astype(stew_dtype)
    fptr.attrs['pc_bits'] = PC_BITS
    fptr.create_dataset(
        "history",
        data=processed_history,
        compression='gzip',
        compression_opts=9,
    )
    return fptr

def get_work_items():
    work_items = []    
    for benchmark in TARGET_BENCHMARKS:
        hard_brs = common.read_hard_brs(benchmark, HARD_BRS_FILE)
        traces_dir = '{}/{}'.format(PATHS['branch_traces_dir'], benchmark)
        datasets_dir = '{}/{}'.format(PATHS['ml_datasets_dir'], benchmark)
        os.makedirs(datasets_dir, exist_ok=True)
        for inp_info in BENCHMARKS_INFO[benchmark]['inputs']:
            for simpoint_info in inp_info['simpoints']:
                file_basename = '{}_{}_simpoint{}'.format(
                    benchmark, inp_info['name'], simpoint_info['id'])
                trace_path = '{}/{}_brtrace.bz2'.format(
                    traces_dir, file_basename)
                dataset_path = '{}/{}_dataset.hdf5'.format(
                    datasets_dir, file_basename)
                
                if os.path.exists(dataset_path): continue
                work_items.append((trace_path, dataset_path, hard_brs))
    return work_items


def gen_dataset(trace_path, dataset_path, hard_brs):
    print('reading file', trace_path)
    pcs, directions = read_branch_trace(trace_path)

    print('Creating output file', dataset_path)
    fptr = create_new_dataset(dataset_path, pcs, directions)

    for br_pc in hard_brs:
        print('processing branch {}'.format(hex(br_pc)))
        #find indicies of hard branches
        trace_br_indices = np.argwhere(pcs == br_pc).squeeze(axis=1)
        fptr.create_dataset(
            'br_indices_{}'.format(hex(br_pc)),
            data=trace_br_indices,
            compression='gzip',
            compression_opts=9,
        )
        num_taken = np.count_nonzero(
            np.bitwise_and(pcs == br_pc, directions == 1))
        num_not_taken = np.count_nonzero(
            np.bitwise_and(pcs == br_pc, directions == 0))
        fptr.attrs['num_taken_{}'.format(hex(br_pc))] = num_taken
        fptr.attrs['num_not_taken_{}'.format(hex(br_pc))] = num_not_taken


def main():
    work_items = get_work_items()
    with multiprocessing.Pool(NUM_THREADS) as pool:
        pool.starmap(gen_dataset, work_items)


if __name__ == '__main__':
    main()
