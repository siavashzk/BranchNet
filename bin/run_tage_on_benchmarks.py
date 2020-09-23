#!/usr/bin/env python3

import common
from common import PATHS, BENCHMARKS_INFO
import os

TARGET_BENCHMARKS = ['leela']
BINARY_NAME = 'tagescl64'
CONFIG_NAME = 'tagescl64'
NUM_THREADS = 32

def main():
    tage_binary = os.path.dirname(os.path.abspath(__file__)) + '/../build/tage/' + BINARY_NAME
    assert os.path.exists(tage_binary), 'Could not find the TAGE binary at ' + tage_binary

    cmds = []
    for benchmark in TARGET_BENCHMARKS:
        traces_dir = '{}/{}'.format(PATHS['branch_traces_dir'], benchmark)
        stats_dir = '{}/{}/{}'.format(PATHS['tage_stats_dir'], CONFIG_NAME, benchmark)
        os.makedirs(stats_dir, exist_ok=True)
        for inp_info in BENCHMARKS_INFO[benchmark]['inputs']:
            for simpoint_info in inp_info['simpoints']:
                id = simpoint_info['id']
                file_basename = '{}_{}_simpoint{}'.format(benchmark, inp_info['name'], id)
                trace_path = ('{}/{}_brtrace.bz2').format(traces_dir, file_basename)
                stats_path = ('{}/{}_stats.csv').format(stats_dir, file_basename)
                out_file = ('{}/{}_run.out').format(stats_dir, file_basename)

                cmd = '{} {} {} &> {}'.format(tage_binary, trace_path, stats_path, out_file)
                cmds.append(cmd)

    common.run_parallel_commands_local(cmds, NUM_THREADS)


if __name__ == '__main__':
    main()