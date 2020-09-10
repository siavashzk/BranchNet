#!/usr/bin/python3

import subprocess

subprocess.call("cd ../src/tracer; make -j tools", shell=True)