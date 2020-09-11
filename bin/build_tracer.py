#!/usr/bin/python3

import subprocess
import os 

tracer_src_path = os.path.dirname(__file__) + '/../src/tracer'

subprocess.call('cd ' + tracer_src_path + '; make -j', shell=True)