#!/usr/bin/env python3

import argparse
import os
import shlex
import subprocess

TRACER_PINTOOL_PATH = '../build/tracer/tracer.so'

global __args__
def create_parser():
    parser = argparse.ArgumentParser(
        description='Takes a pin ball as input and creates a branch trace')
    parser.add_argument("input_path", type=str, help='Path to the input pinball')
    parser.add_argument("output_path", type=str, help='Path to the output branch trace')
    return parser.parse_args()

def main():
    global __args__
    __args__ = create_parser()
    pin_root = os.environ['PIN_ROOT']

    assert os.path.exists(TRACER_PINTOOL_PATH)
    assert os.path.exists(pin_root + '/pin')
    os.makedirs(os.path.dirname(os.path.abspath(__args__.output_path)), exist_ok=True)

    cmd = ('{pin_root}/pin -xyzzy -virtual_segments 1 -reserve_memory'
           + ' {input_path}.address -t {pintool_path}'
           + ' -replay -replay:basename {input_path}'
           + ' -trace_out_file {output_path}'
           + ' -compressor /usr/bin/bzip2'
           + ' -warmup_instructions 0'
           + ' -- {pin_root}/extras/pinplay/bin/intel64/nullapp').format(
               pin_root=pin_root,
               pintool_path=TRACER_PINTOOL_PATH,
               input_path=__args__.input_path,
               output_path=__args__.output_path,
           )
    subprocess.call(shlex.split(cmd))

if __name__ == '__main__':
    main()