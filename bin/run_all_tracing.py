#!/usr/bin/python3

import common
from common import PATHS, BENCHMARKS_INFO
import os

TARGET_BENCHMARKS = ['leela']

def main():
    cmds = []
    for benchmark in TARGET_BENCHMARKS:
        output_dir = '{}/{}'.format(PATHS['branch_traces_dir'], benchmark)
        os.makedirs(output_dir, exist_ok=True)
        for inp_info in BENCHMARKS_INFO[benchmark]['inputs']:
            for simpoint_info in inp_info['simpoints']:
                id = simpoint_info['id']
                pinball_path = simpoint_info['path']
                trace_path = ('{}/{}_{}_simpoint{}_brtrace.bz2').format(
                    output_dir, benchmark, inp_info['name'], id)

                if os.path.exists(trace_path):
                    continue
                cmd = 'python create_trace.py {} {}'.format(pinball_path, trace_path)
                cmds.append(cmd)

    common.run_parallel_commands_local(cmds)


if __name__ == '__main__':
    main()
