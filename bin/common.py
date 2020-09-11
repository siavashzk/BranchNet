import multiprocessing
import os
import shlex
import subprocess
import yaml

__env_dir__ = os.path.dirname(__file__) + '/../environment_setup'
__paths_file__ = __env_dir__ + '/paths.yaml'
__benchmarks_file__ = __env_dir__ + '/benchmarks.yaml'

assert os.path.exists(__paths_file__), (
  ('Expecting a paths.yaml file at {}. You have to create one following '
   'the format of paths_example.yaml in the same directory').format(__paths_file__))
assert os.path.exists(__benchmarks_file__), (
  ('Expecting a benchmarks.yaml file at {}. You have to create one following '
   'the format of benchmarks_example.yaml in the same directory').format(__benchmarks_file__))

with open(__paths_file__) as f:
  PATHS = yaml.safe_load(f)

with open(__benchmarks_file__) as f:
  BENCHMARKS_INFO = yaml.safe_load(f)

def run_cmd_using_shell(cmd):
  print('Running cmd:', cmd)
  subprocess.call(shlex.split(cmd))

def run_parallel_commands_local(cmds, num_threads=None):
  with multiprocessing.Pool(num_threads) as pool:
    pool.map(run_cmd_using_shell, cmds)