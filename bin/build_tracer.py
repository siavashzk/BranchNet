#!/usr/bin/env python3

import subprocess
import os 

tracer_src_path = os.path.dirname(os.path.abspath(__file__)) + '/../src/tracer'

subprocess.call('cd ' + tracer_src_path + '; make -j', shell=True)