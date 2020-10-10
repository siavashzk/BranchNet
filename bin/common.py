import csv
import glob
import multiprocessing
import os
import re
import subprocess
import yaml

SIMPOINT_LENGTH = 200000000

__env_dir__ = os.path.dirname(__file__) + '/../environment_setup'
__paths_file__ = __env_dir__ + '/paths.yaml'
__benchmarks_file__ = __env_dir__ + '/benchmarks.yaml'
__ml_input_partitions__ = __env_dir__ + '/ml_input_partitions.yaml'


assert os.path.exists(__paths_file__), (
  ('Expecting a paths.yaml file at {}. You have to create one following '
   'the format of paths_example.yaml in the same directory').format(__paths_file__))
assert os.path.exists(__benchmarks_file__), (
  ('Expecting a benchmarks.yaml file at {}. You have to create one following '
   'the format of benchmarks_example.yaml in the same directory').format(__benchmarks_file__))
assert os.path.exists(__ml_input_partitions__), (
  ('Expecting an ml_input_partitions.yaml file at {}. You have to create one following '
   'the format of ml_input_partitions_example.yaml in the same directory').format(__ml_input_partitions__))

with open(__paths_file__) as f:
  PATHS = yaml.safe_load(f)

with open(__benchmarks_file__) as f:
  BENCHMARKS_INFO = yaml.safe_load(f)

with open(__ml_input_partitions__) as f:
  ML_INPUT_PARTIONS = yaml.safe_load(f)


def run_cmd_using_shell(cmd):
  print('Running cmd:', cmd)
  subprocess.call(cmd, shell=True)


def run_parallel_commands_local(cmds, num_threads=None):
  with multiprocessing.Pool(num_threads) as pool:
    pool.map(run_cmd_using_shell, cmds)


class BranchStats:
  def __init__(self):
    self.correct = 0
    self.incorrect = 0
    self.total = 0
    self.instructions = SIMPOINT_LENGTH 
    self.accuracy = 0
    self.mpki = 0

  def InitWithCorrect(self, correct, total, instructions=None):
    if (total == 0):
      self.__init__()
    else:
      if (instructions != None): self.instructions = instructions
      self.correct = correct
      self.total = total
      self.incorrect = total - correct
      self.accuracy  = correct / total if total != 0 else 0.0
      self.mpki = (float(self.incorrect) / self.instructions) * 1000

  def InitWithAccuracy(self, accuracy, total):
    if (total == 0):
      self.__init__()
    else:
      self.accuracy  = accuracy
      self.total = total
      self.correct = accuracy * total
      self.incorrect = total - self.correct
      self.mpki = (float(self.incorrect) / self.instructions) * 1000

  def ApplyMPKIReduction(self, mpki_reduction):
    copy = BranchStats()
    copy.total = self.total
    copy.mpki = self.mpki - mpki_reduction
    copy.inccorect = (copy.mpki / 1000 ) * self.instructions
    copy.correct = copy.total - copy.incorrect
    copy.accuracy  = copy.correct / copy.total if copy.total != 0 else 0.0
    return copy


class AggregateBranchStats:
  def __init__(self, num_simpoints):
    self.weighted_stats = BranchStats()
    self.unweighted_stats = BranchStats()
    self.region_stats = []
    for i in range(num_simpoints):
      self.region_stats.append(BranchStats())

  def finalize_stats(self, weights):
    assert len(weights) == len(self.region_stats) 
    weighted_correct = 0
    weighted_total = 0
    weighted_instructions = 0
    unweighted_correct = 0
    unweighted_total = 0
    unweighted_instructions = 0
    for simpoint_region in range(len(weights)):
      entry = self.region_stats[simpoint_region]
      weighted_correct += weights[simpoint_region] * entry.correct
      weighted_total += weights[simpoint_region] * entry.total
      weighted_instructions += weights[simpoint_region] * entry.instructions
      unweighted_correct += entry.correct
      unweighted_total += entry.total
      unweighted_instructions += entry.instructions
    self.weighted_stats.InitWithCorrect(weighted_correct, weighted_total, weighted_instructions)
    self.unweighted_stats.InitWithCorrect(unweighted_correct, unweighted_total, unweighted_instructions)


def get_simpoint_info(benchmark, inp):
    inps_info = BENCHMARKS_INFO[benchmark]['inputs']
    for inp_info in inps_info:
        if inp_info['name'] == inp:
            return inp_info['simpoints']
    else:
        assert False, 'Did not find information about benchmark {}, input {}'.format(benchmark, inp)


def get_simpoint_weights(benchmark, inp):
    simpoint_info = get_simpoint_info(benchmark, inp)
    return [simpoint['weight'] for simpoint in simpoint_info]


def update_tage_stats(tage_stats, stats_file, num_simpoints, simpoint_id):
    PC_COLUMN_IDX = 0
    CORRECT_COLUMN_IDX = 3
    TOTAL_COLUMN_IDX = 4

    with open(stats_file) as f:
        reader = csv.reader(f, delimiter=',')
        headers = next(reader)
        assert headers[PC_COLUMN_IDX] == 'Branch PC'
        assert headers[CORRECT_COLUMN_IDX] == 'Correct Predictions'
        assert headers[TOTAL_COLUMN_IDX] == 'Total'

        for cols in reader:
            br = (-1 if cols[PC_COLUMN_IDX] == "aggregate"
                     else int(cols[PC_COLUMN_IDX], 16))
            correct = int(cols[CORRECT_COLUMN_IDX]) 
            total = int(cols[TOTAL_COLUMN_IDX]) 
            if br not in tage_stats:
                tage_stats[br] = AggregateBranchStats(num_simpoints)
            tage_stats[br].region_stats[simpoint_id].InitWithCorrect(
                correct, total)


def read_tage_stats(tage_config_name, ml_benchmark, inp, hard_brs_tag=None):
    spec_benchmark = ML_INPUT_PARTIONS[ml_benchmark]['spec_name']
    weights = get_simpoint_weights(spec_benchmark, inp)
    num_simpoints = len(weights)
    stats_dir = '{}/{}{}/{}'.format(
        PATHS['tage_stats_dir'], 
        ('/noalloc_for_hard_brs_' + hard_brs_tag) if hard_brs_tag else '',
        tage_config_name,
        spec_benchmark)

    tage_stats = {}
    for simpoint in get_simpoint_info(spec_benchmark, inp):
        stats_file = ('{}/{}_{}_simpoint{}_stats.csv').format(
            stats_dir, spec_benchmark, inp, simpoint['id'])
        update_tage_stats(tage_stats, stats_file, num_simpoints, simpoint['id'])

    for stats in tage_stats.values():
        stats.finalize_stats(weights)
    return tage_stats


def read_hard_brs(benchmark, name):
    filepath = '{}/{}_{}'.format( PATHS['hard_brs_dir'], benchmark, name)
    with open (filepath) as f:
        return [int(x,16) for x in f.read().splitlines()]


def update_cnn_stats(cnn_stats, tage_stats, results_file, num_simpoints, target_inp):
    br_str, _ = os.path.splitext(os.path.basename(results_file))
    br = int(br_str, 16)

    if br not in tage_stats: return

    with open (results_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            m = re.search('_(.*)_simpoint([0-9]+)_dataset', row[0])
            assert m is not None
            inp = m.group(1)
            simpoint_region = int(m.group(2))
            accuracy = float(row[1])

            if inp != target_inp: continue

            if br not in cnn_stats:
              cnn_stats[br] = AggregateBranchStats(num_simpoints)

            cnn_stats[br].region_stats[simpoint_region].InitWithAccuracy(
                accuracy, tage_stats[br].region_stats[simpoint_region].total)


def read_cnn_stats(ml_benchmark, experiment_name, inp, tage_stats):
    spec_benchmark = ML_INPUT_PARTIONS[ml_benchmark]['spec_name']
    weights = get_simpoint_weights(spec_benchmark, inp)
    num_simpoints = len(weights)
    results_dir = '{}/{}/{}/results'.format(PATHS['experiments_dir'], experiment_name, ml_benchmark)

    cnn_stats = {}
    results_files = glob.glob(results_dir + '/*.csv')
    assert results_files
    for results_file in results_files:
        update_cnn_stats(cnn_stats, tage_stats, results_file, num_simpoints, inp)

    for stats in cnn_stats.values():
        stats.finalize_stats(weights)
    cnn_stats[-1] = AggregateBranchStats(num_simpoints)
    return cnn_stats