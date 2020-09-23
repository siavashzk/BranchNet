#!/usr/bin/env python3

import common
from common import PATHS, BENCHMARKS_INFO
import os

TRACER_PINTOOL_PATH = '../build/tracer/tracer.so'
TARGET_BENCHMARKS = ['deepsjeng']
NUM_THREADS = 24

def get_run_cmd(pinball_path, trace_path):
    pin_root = os.environ['PIN_ROOT']
    assert os.path.exists(TRACER_PINTOOL_PATH)
    assert os.path.exists(pin_root + '/pin')

    os.makedirs(os.path.dirname(os.path.abspath(trace_path)), exist_ok=True)

    return ('{pin_root}/pin -xyzzy -virtual_segments 1 -reserve_memory'
            + ' {input_path}.address -t {pintool_path}'
            + ' -replay -replay:basename {input_path}'
            + ' -trace_out_file {output_path}'
            + ' -compressor /usr/bin/bzip2'
            + ' -warmup_instructions 0'
            + ' -- {pin_root}/extras/pinplay/bin/intel64/nullapp').format(
                pin_root=pin_root,
                pintool_path=TRACER_PINTOOL_PATH,
                input_path=pinball_path,
                output_path=trace_path,
            )


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
                cmds.append(get_run_cmd(pinball_path, trace_path))

    common.run_parallel_commands_local(cmds, NUM_THREADS)


if __name__ == '__main__':
    main()
