"""
Pytorch DataLoader for reading instances of hard to predict branches and their
histories from a collection of hdf5 trace files.
"""
from __future__ import print_function
from __future__ import division


from multiprocessing import Lock
import numpy as np
from torch.utils.data import Dataset

import h5py

def get_pos_weight(traces, br_pc):
  num_taken = 0
  num_not_taken = 0
  num_taken_attr = 'num_taken_{}'.format(hex(br_pc))
  num_not_taken_attr = 'num_not_taken_{}'.format(hex(br_pc))
  for trace in traces:
    file_ptr = h5py.File(trace, 'r')
    if (num_taken_attr not in file_ptr.attrs or
        num_not_taken_attr not in file_ptr.attrs):
      print('Warning: traces files do not contain branch bias information')
      return 1.0
    num_taken += file_ptr.attrs[num_taken_attr]
    num_not_taken += file_ptr.attrs[num_not_taken_attr]
  return num_not_taken / num_taken
        


def remove_incomplete_histories(br_indices, history_length):
  """Filters out instances of a branch with incomplete histories.

  Args:
    br_indices: a numpy array of indices of the occurances of a branch
      sorted in increasing order.
    history_length: the minimum expected history length that should be
      available for each instance of the target branch.

  Returns:
    a sorted numpy array of indices, each is guaranteed to meet the history
      length requirement.
  """
  if br_indices.size != 0:
    first_valid_idx = 0
    while (first_valid_idx < len(br_indices) and
           br_indices[first_valid_idx] < history_length):
      first_valid_idx += 1
    br_indices = br_indices[first_valid_idx:]

  return br_indices


def get_branch_indices(hdf5_file_ptr, br_pc):
  """Reads the indices of a given branch from an hdf5 file into a numpy array.

  It assumes that the hdf5 already contains the branch indices with a dataset
  name of br_indices_{Branch PC in hexadecimal}.

  Args:
    hdf5_file_ptr: h5py File object that is already opened for reading.
    br_pc: the PC of the target branch (as an int).

  Returns:
    A numpy array of the branch indices.

  Raises:
    An Exception if the branch indices dataset is not found in the hdf5 file.
  """
  dataset_name = 'br_indices_'+hex(br_pc)
  if dataset_name not in hdf5_file_ptr.keys():
    raise Exception(
        'hdf5 file {} does not contain the indices'
        ' of occurances of branch {}'.format(
            hdf5_file_ptr.filename, hex(br_pc)))
  return hdf5_file_ptr[dataset_name][:]


def preprocess_history(history, pc_bits, pc_hash_bits, hash_dir_with_pc, dtype):
  """Preprocesses a history array by hashing the PC in each history element.

  It takes the least significant bits of the PC (controlled by argument
    pc_bits) and hashes them into potentially fewer number of bits (controlled
    by pc_hash_bits argument) and concatenates the PC with direction bit. After
    the hashing, it converts the array to the desired numpy data type.

  Args:
    history: history as a numpy array.
    pc_bits : The number of least significant bits of the PC to use in
      the history.
    pc_hash_bits : The width of PC hash that is produced for input
    dtype: the desired output numpu data type.

  Returns:
    a numpy array containing the hashed history with the desired data type.
  """
  pc_mask = (1 << (1 + pc_bits)) - 1
  np.bitwise_and(history, pc_mask, out=history)
  if hash_dir_with_pc:
    if pc_hash_bits < (pc_bits + 1):
      unprocessed_bits = pc_bits + 1 - pc_hash_bits
      pc_hash_mask = ((1 << pc_hash_bits) - 1)
      shift_count = 1
      temp = np.empty_like(history)
      while unprocessed_bits > 0:
        np.right_shift(history, shift_count * pc_hash_bits, out=temp)
        np.bitwise_and(temp, pc_hash_mask, out=temp)
        np.bitwise_xor(history, temp, out=history)
        shift_count += 1
        unprocessed_bits -= pc_hash_bits
      np.bitwise_and(history, pc_hash_mask, out=history)
  else:
    if pc_hash_bits < pc_bits:
      unprocessed_bits = pc_bits - pc_hash_bits
      pc_hash_mask = ((1 << pc_hash_bits) - 1) << 1
      shift_count = 1
      temp = np.empty_like(history)
      while unprocessed_bits > 0:
        np.right_shift(history, shift_count * pc_hash_bits, out=temp)
        np.bitwise_and(temp, pc_hash_mask, out=temp)
        np.bitwise_xor(history, temp, out=history)
        shift_count += 1
        unprocessed_bits -= pc_hash_bits

      stew_mask = (1 << (pc_hash_bits + 1)) - 1
      np.bitwise_and(history, stew_mask, out=history)
  return history.astype(dtype)

class TraceFileAccessor():
  """Wrapper for random accesses to an hdf5 trace file.

  Attributes:
    history_length (int): Length of history to read for each branch instance.
    pc_bits (int): The number of least significant bits of the PC to use
      in the history.
    pc_hash_bits (int): The width of PC hash that is produced for input
    path (str): path to the trace file.
    keep_file_open (bool): Flag identifying that the File object should be kept
      open throughout the life of the object.
    br_indices (:obj:`numpy.array`): List of all indices of all occurances of
      the target branch in the history.
    file_ptr (:obj:`h5py.File`, optional): File object of the opened trace file.
    in_mem_history (:obj:`numpy.array`, optional): Preprocessed branch trace
      read from the file into main memory.
  """

  def __init__(self, trace_path, *,
               br_pc, history_length, pc_bits, pc_hash_bits, hash_dir_with_pc, in_mem,
               keep_file_open):
    """Initializes class attributes.

    Optionally, reads and preprocesses the whole branch trace into memory.

    Args:
      br_pc (int): the PC of the target branch.

    See class attributes for description of other arguments.
    """
    assert pc_bits >= pc_hash_bits
    self.history_length = history_length
    self.stew_mask = (1 << (pc_bits + 1)) - 1
    self.pc_bits = pc_bits
    self.pc_hash_bits = pc_hash_bits
    self.hash_dir_with_pc = hash_dir_with_pc
    self.path = trace_path
    self.br_indices = None
    self.keep_file_open = keep_file_open
    self.file_ptr = None
    self.in_mem_history = None

    self.file_ptr = h5py.File(trace_path, 'r')

    self.br_indices = get_branch_indices(self.file_ptr, br_pc)
    self.br_indices = remove_incomplete_histories(
        self.br_indices, history_length)

    if in_mem and len(self.br_indices) > 0:
      if pc_hash_bits <= 7:
        in_mem_dtype = np.uint8
      else:
        assert pc_hash_bits <= 15
        in_mem_dtype = np.uint16
      self.in_mem_history = preprocess_history(
          self.file_ptr['history'][:], self.pc_bits,
          self.pc_hash_bits, self.hash_dir_with_pc, in_mem_dtype)

    if not self.keep_file_open:
      self.file_ptr.close()
      self.file_ptr = None


  def get_history(self, occurance_idx):
    """Gets the chunk of the trace containing the target branch and its history.

    Returns:
      A numpy array taken from the branch history including an occurance of
        the target branch and its history.
    """
    assert 0 <= occurance_idx < len(self.br_indices)

    idx = self.br_indices[occurance_idx]
    if self.in_mem_history is not None:
      chunk = self.in_mem_history[idx - self.history_length : idx + 1]
      chunk = chunk.astype(np.int64)
      markers, = np.nonzero(np.logical_or(chunk == 0x80, chunk == 0x81))
      if len(markers):
        closest_marker = markers[-1]
        chunk[0:closest_marker] = 0
    else:
      if not self.keep_file_open:
        self.file_ptr = h5py.File(self.path, 'r')

      chunk = self.file_ptr['history'][idx - self.history_length : idx + 1]
      chunk = preprocess_history(
          chunk, self.pc_bits, self.pc_hash_bits, np.int64)

      if not self.keep_file_open:
        self.file_ptr.close()
        self.file_ptr = None

    return chunk

  def num_instances(self):
    """Gets the number of instances of the branch in this trace.

    Returns:
      The length of self.br_indices, which equals to the number of
        occurances of the target branch.
    """
    return len(self.br_indices)

  def __del__(self):
    if self.file_ptr is not None:
      self.file_ptr.close()



class BranchDataset(Dataset):
  """A pytorch dataset class that for reading branch traces from hdf5 traces.

  A dataset class can be used with pytorch Dataloader to feed
    branch traces into a pytorch neural network model. This class provides a
    unified logical view for all the given trace files and provides a random
    accessor from the combined set of traces.

  Attributes:
    trace_accessors : A list of TraceFileAccessor objects for each trace file.
    locks (optional): A list of multiprocessing Lock objects associated with
      each trace file.
    trace_end_instances: A list of integers, each signifying the largest global
      index covered by each file. On a random access, this list is searched
      to find which file contains the target branch.
    use_lock: A boolean flag that turns/on off use of locks. Should be set to
      True if TraceFileAccessor implementation is not thread-safe.
  """

  def __init__(self, trace_paths, *,
               br_pc, history_length, pc_bits, pc_hash_bits, hash_dir_with_pc,
               in_mem=True, use_lock=False):
    """Creates a TraceFileAccessor for each trace_path and sets
    trace_end_instances to provide a unified global view to all traces.

    Args:
      trace_paths (str): A list of file paths of the trace files.
      br_pc (int): the PC of the target branch.
      history_length (int): Length of history to read for each branch instance.
      pc_bits (int): The number of least significant bits of the PC to use
        in the history.
      pc_hash_bits (int): The width of PC hash that is produced for input
      in_mem (bool): Set the trace accessors to preprocess the traces and load
        them into the trace file. If False, trace accessors read the history
        chunk from the file system for each random access.
      use_lock (bool) See class attribute use_lock.
    """

    #By inserting a dummy entry into the list of items associated with
    #the traces, we make the logic for __getitemm__() easier.
    self.trace_accessors = [None]
    self.locks = [None]
    self.trace_end_instances = np.array([-1])
    self.use_lock = use_lock

    for i, trace_path in enumerate(trace_paths):
      trace_accessor = TraceFileAccessor(
          trace_path,
          br_pc=br_pc,
          history_length=history_length,
          pc_bits=pc_bits,
          pc_hash_bits=pc_hash_bits,
          hash_dir_with_pc=hash_dir_with_pc,
          in_mem=in_mem,
          keep_file_open=not in_mem)

      self.trace_accessors.append(trace_accessor)
      self.trace_end_instances = np.append(
          self.trace_end_instances,
          self.trace_end_instances[i] + trace_accessor.num_instances())
      if use_lock:
        self.locks.append(Lock())

  def __len__(self):
    return self.trace_end_instances[-1] + 1

  def __getitem__(self, idx):
    file_idx = np.searchsorted(self.trace_end_instances, idx)
    internal_idx = idx - self.trace_end_instances[file_idx - 1] - 1

    if self.use_lock:
      self.locks[file_idx].acquire()
    history_chunk = self.trace_accessors[file_idx].get_history(internal_idx)
    if self.use_lock:
      self.locks[file_idx].release()

    inputs = history_chunk[:-1] #Last element is the target branch itself.
    label = (history_chunk[-1] & 1).astype(np.float32)
    return (inputs, label)
