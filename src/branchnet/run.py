"""
Main file for the scripts to train and evaluate a CNN branch predictor
"""

import argparse
import copy
import os
import numpy as np
import timeit
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import dataset_loader  
from dataset_loader import BranchDataset
from model import BranchNet, BranchNetTrainingPhaseKnobs


def create_parser():
  """This function defines the command-line interface and parses the
    commandline accordingly.

  Returns:
    A populated namespace of all arguments.
  """
  parser = argparse.ArgumentParser(
      description='Pytorch program for training or evaluating a CNN '
                  'branch predictor')

  parser.add_argument('-trtr', '--training_traces',
                      nargs='+',
                      required=True,
                      help=('Paths of the preprocessed hdf5 traces to be used '
                            'for the training set'))
  parser.add_argument('-vvtr', '--validation_traces',
                      nargs='+',
                      help=('Paths of the preprocessed hdf5 traces to be used '
                            'for the validation set'))
  parser.add_argument('-evtr', '--evaluation_traces',
                      nargs='+',
                      required=True,
                      help=('Paths of the preprocessed hdf5 traces to be used '
                            'for final evaluation'))

  parser.add_argument('-mode', '--training_mode',
                      default='float',
                      choices=['float', 'mini', 'tarsa'],
                      help='Mode of training')
  parser.add_argument('--br_pc',
                      type=lambda x: int(x, 16),
                      required=True,
                      help='The PC of the target hard to predict branch')
  parser.add_argument('--workdir',
                      default=os.getcwd(),
                      help='Path to the working directory, used for storing '
                           'checkpoints, logs, and results files')
  parser.add_argument('-c', '--config_file',
                      required=True,
                      help='Name of the config file to use (should be in the '
                           'work directory)')

  parser.add_argument('-batch', '--batch_size',
                      type=int,
                      default=512,
                      help='Training/Inference Batch Size')
  parser.add_argument('-bsteps', '--base_training_steps',
                      type=int,
                      nargs='*',
                      default=[],
                      help='Number of steps for each training interval with '
                           'exponential decay on an unpruned model without '
                           'group lasso.')
  parser.add_argument('-fsteps', '--fine_tuning_training_steps',
                      type=int,
                      nargs='*',
                      default=[],
                      help='Number of steps for each training interval with '
                           'exponential decay for fine-tuning after '
                           'convolution layer is hardenend.')
  parser.add_argument('-gsteps', '--group_lasso_training_steps',
                      type=int,
                      nargs='*',
                      default=[],
                      help='Number of steps for each training interval with '
                           'exponential decay on an unpruned model with '
                           'group lasso.')
  parser.add_argument('-psteps', '--pruned_training_steps',
                      type=int,
                      nargs='*',
                      default=[],
                      help='Number of steps for each training interval with '
                           'exponential decay on a pruned model.')
  parser.add_argument('-lr', '--learning_rate',
                      type=float,
                      default=0.002,
                      help='Initial learning rate')
  parser.add_argument('-gcoeff', '--group_lasso_coeff',
                      type=float,
                      default=0.0,
                      help='Group Lasso loss term coefficient')
  parser.add_argument('-rcoeff', '--fc_regularization_coeff',
                      type=float,
                      default=0.0,
                      help='Fully-connected layers regularization coefficient')

  parser.add_argument('--cuda_device',
                      type=int,
                      default=0,
                      help='Cuda device number (-1 means cpu)')
  parser.add_argument('--log_progress',
                      action='store_true',
                      help='Log training progress')
  parser.add_argument('--log_validation',
                      action='store_true',
                      help='Log validation loss (NOP if log_progress is not set)')

  return parser.parse_args()

__args__ = create_parser()

class LossLogger():
  def __init__(self):
    self.training_loss = []
    self.group_lasso_loss = []
    self.fc_reg_loss = []
    self.learning_rate = []
    self.validation_steps = []
    self.validation_loss = []
    self.validation_accuracy = []

  def log_training(self, prediction_loss, group_lasso_loss,
                   fc_reg_loss, learning_rate):
    self.training_loss.append(prediction_loss)
    self.group_lasso_loss.append(group_lasso_loss)
    self.fc_reg_loss.append(fc_reg_loss)
    self.learning_rate.append(learning_rate)

  def log_validation(self, validation_loss, validation_accuracy):
    self.validation_steps.append(len(self.training_loss) - 1)
    self.validation_loss.append(validation_loss)
    self.validation_accuracy.append(validation_accuracy)

  def validation_is_behind(self):
    return ((len(self.validation_steps) == 0) or
            (self.validation_steps[-1] < len(self.training_loss) - 1))

  def plot_loss(self, filename):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10,20))
    ax1.plot(self.training_loss, label='Training loss')
    ax1.plot(self.validation_steps, self.validation_loss, label='Validation loss')
    ax1.set_title('Loss')
    ax1.legend()
    ax2.plot(self.validation_steps, self.validation_accuracy)
    ax2.set_title('Validation Accuracy')
    ax1.set_ylim(0, 1)
    ax3.plot(self.group_lasso_loss, label='Group Lasso Regulariozation')
    ax3.plot(self.fc_reg_loss, label='Fully-connected Regulariozation')
    ax3.set_title('Regularization Losses')
    ax3.legend()
    ax4.plot(self.learning_rate)
    ax4.set_title('Learning Rate')
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)

class ModelWrapper():
  """A wrapper around a branch predictor model for training and evaluation.

  Attributes:
    model: A Pytorch neural network module.
    training_traces: Paths of the traces used for the training set.
    validation_traces: Paths of the traces used for the validation set.
    test_traces: Paths of the traces used for the test set.
    br_pc: the PC of the target branch (as an int).
    branchnet_config: branchnet configuration dictionary.
    batch_size: Training and Inference batch size.
    dataloader: The active pytorch DataLoader for branchnet (could be None).
    dataloader_traces: The set of branch traces in the active dataloader.
  """
  def __init__(self, model, *,
               br_pc, branchnet_config, batch_size, cuda_device, log_progress):
    """Simply initializes class attributes based on the constructor arguments
    """
    self.model = model
    self.br_pc = br_pc
    self.branchnet_config = branchnet_config
    self.batch_size = batch_size
    self.device = torch.device('cpu') if cuda_device == -1 else torch.device('cuda:'+str(cuda_device))
    self.logger = LossLogger() if log_progress else None
    self.model.to(self.device)
    self.dataloader_dict = {}
    self.criterion = nn.BCEWithLogitsLoss(reduction='mean')

  def get_dataloader(self, traces, eval=False):
    dict_key = ' '.join(traces)
    if dict_key in self.dataloader_dict:
      return self.dataloader_dict[dict_key]

    # need to have enough history for the largest slice
    history_length = max(self.branchnet_config['history_lengths'])
    if any(self.branchnet_config['shifting_pooling']):
      # need to add extra history to support random shifting
      history_length += max(self.branchnet_config['pooling_widths'])

    dataset = BranchDataset(
        traces,
        br_pc=self.br_pc,
        history_length=history_length,
        pc_bits=self.branchnet_config['pc_bits'],
        pc_hash_bits=self.branchnet_config['pc_hash_bits'],
        hash_dir_with_pc=self.branchnet_config['hash_dir_with_pc'])
    if len(dataset) > 0:
      dataloader = DataLoader(
          dataset, batch_size=self.batch_size,
          shuffle=not eval, num_workers=6, pin_memory=True)
    else:
      dataloader = None

    self.dataloader_dict[dict_key] = dataloader
    return dataloader
    
  def get_loss_values(self, outs, labels, group_lasso_coeff,
                      fc_regularization_coeff):
    loss = self.criterion(outs, labels)
    prediction_loss_value = loss.item()

    group_lasso_loss_value = 0

    if group_lasso_coeff > 0:
      group_lasso_loss = group_lasso_coeff * self.model.group_lasso_loss()
      group_lasso_loss_value = group_lasso_loss.item()
      loss += group_lasso_loss

    fc_reg_loss_value = 0
    if fc_regularization_coeff > 0:
      fc_reg_loss = fc_regularization_coeff * self.model.fc_weights_l1_loss()
      fc_reg_loss_value = fc_reg_loss.item()
      loss += fc_reg_loss

    return (loss, prediction_loss_value,
            group_lasso_loss_value, fc_reg_loss_value)


  def train(self, training_traces, training_steps, learning_rate,
            group_lasso_coeff, fc_regularization_coeff):
    """Train one epoch using the training set"""
    self.model.train()
    training_set_loader = self.get_dataloader(training_traces)

    optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.2)

    for num_steps in training_steps:
      print('Training for {} steps with learning rate {}'.format(
          num_steps, scheduler.get_lr()[0]))
      step = 0
      while step < num_steps:
        for inps, labels in training_set_loader:
          outs = self.model(inps.to(self.device))
          (loss, prediction_loss_value,
           group_losso_loss_value, fc_reg_loss_value) = self.get_loss_values(
               outs, labels.to(self.device), group_lasso_coeff,
               fc_regularization_coeff)

          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          if self.logger is not None:
            self.logger.log_training(prediction_loss_value, 
                                     group_losso_loss_value,
                                     fc_reg_loss_value,
                                     scheduler.get_lr()[0])

            if (step + 1) % 500 == 0:
              #print('Evaluating the validation set')
              #if __args__.validation_traces:
              #  corrects, total, loss = self.eval(__args__.validation_traces)
              #  self.model.train()
              #  self.logger.log_validation(loss, corrects / total * 100)
              #else:
              #  self.logger.log_validation(0, 0)
              self.logger.plot_loss('visual_log_{}.pdf'.format(hex(self.br_pc)))

          step += 1
          if step >= num_steps:
            break

      scheduler.step()
      if self.logger is not None:
        if self.logger.validation_is_behind():
          if __args__.validation_traces and __args__.log_validation:
            print('Evaluating the validation set')
            corrects, total, loss = self.eval(__args__.validation_traces)
            self.model.train()
            self.logger.log_validation(loss, corrects / total * 100)
          else:
            self.logger.log_validation(0, 0)
        self.logger.plot_loss('{}/visual_logs/{}.pdf'.format(__args__.workdir, hex(self.br_pc)))

  def eval(self, trace_list):
    """Evaluate the predictor on the traces passed.
    """

    self.model.eval()
    test_set_loader = self.get_dataloader(trace_list, eval=True)
    #loader could be None if no branches of br_pc is found in the traces
    corrects = 0
    total = 0
    total_loss = 0
    if test_set_loader is not None: 
      for inps, labels in test_set_loader:
        inps = inps.to(self.device)
        labels = labels.to(self.device)
        outs = self.model(inps)
        _, loss_value, _, _ = self.get_loss_values(
            outs, labels, 0.0, 0.0)
        predictions = outs > 0
        targets = labels > 0.5
        corrects += (predictions == targets).sum().item()
        total += len(predictions)
        total_loss += loss_value
      total_loss = total_loss / len(test_set_loader)
    return corrects, total, total_loss
    

  def load_checkpoint(self, checkpoint_path):
    """ Loads a checkpoint file to initialize the model state
    """
    self.model.load_state_dict(torch.load(
        checkpoint_path, map_location=lambda storage, loc: storage))

  def save_checkpoint(self, checkpoint_path):
    """ Saves the model state into a checkpoint file
    """
    torch.save(self.model.state_dict(), checkpoint_path)


def create_full_precision_branchnet_model(config):
  training_phase_knobs = BranchNetTrainingPhaseKnobs()
  model = BranchNet(config, training_phase_knobs)
  model_wrapper = ModelWrapper(
      model,
      br_pc=__args__.br_pc,
      branchnet_config=config,
      batch_size=__args__.batch_size,
      cuda_device=__args__.cuda_device,
      log_progress=__args__.log_progress)
  return model_wrapper


def convert_convolutions_to_luts(model_wrapper):
  full_precision_model = model_wrapper.model
  training_phase_knobs = copy.deepcopy(
      full_precision_model.training_phase_knobs)
  training_phase_knobs.lut_convolution = True
  model = BranchNet(full_precision_model.config, training_phase_knobs).to(
      model_wrapper.device)
  model.load_convolution_luts(full_precision_model)
  model_wrapper.model = model
  return model_wrapper


def prune_convolution_filters(model_wrapper):
  useful_filters = find_most_useful_filters(model_wrapper)
  unpruned_model = model_wrapper.model
  training_phase_knobs = copy.deepcopy(
      unpruned_model.training_phase_knobs)
  training_phase_knobs.prune_filters = True
  model = BranchNet(unpruned_model.config, training_phase_knobs).to(
      model_wrapper.device)
  model.copy_from_other_model(unpruned_model)
  model.setup_conv_pruning_masks(useful_filters)
  model_wrapper.model = model
  return model_wrapper


def change_training_phase_knobs(model_wrapper, new_knobs_dict):
  base_model = model_wrapper.model
  training_phase_knobs = copy.deepcopy(base_model.training_phase_knobs)
  for key in new_knobs_dict:
    setattr(training_phase_knobs, key, new_knobs_dict[key])
  model = BranchNet(base_model.config, training_phase_knobs).to(
      model_wrapper.device)
  model.copy_from_other_model(base_model)
  model_wrapper.model = model
  return model_wrapper


def load_checkpoint_or_train_and_create_checkpoint(
    model_wrapper, checkpoint_path, training_steps, group_lasso_coeff,
    fc_regularization_coeff):
  if (os.path.isfile(checkpoint_path)):
    print('Loading Checkpoint at {}'.format(checkpoint_path))
    model_wrapper.load_checkpoint(checkpoint_path)
    return

  if training_steps:
    model_wrapper.train(__args__.training_traces,
                        training_steps,
                        __args__.learning_rate,
                        0.0,
                        fc_regularization_coeff)
    if group_lasso_coeff > 0:
      model_wrapper.train(__args__.training_traces,
                          training_steps,
                          __args__.learning_rate,
                          group_lasso_coeff,
                          fc_regularization_coeff)
  print('Saving Checkpoint at {}'.format(checkpoint_path))
  model_wrapper.save_checkpoint(checkpoint_path)


def full_precision_training_phase(
    model_wrapper, *,
    group_lasso_coeff=__args__.group_lasso_coeff,
    fc_regularization_coeff=0.0):
  print('=======================    Full-precision Training Phase   '
        '=========================')
  checkpoint_path = '{}/checkpoints/base_{}_checkpoint.pt'.format(
      __args__.workdir, hex(__args__.br_pc))
  load_checkpoint_or_train_and_create_checkpoint(
      model_wrapper, checkpoint_path, __args__.base_training_steps,
      group_lasso_coeff, fc_regularization_coeff)


def fine_tune_the_model(model_wrapper, phase_message, checkpoint_prefix, *,
    fc_regularization_coeff=0.0):
  print('=======================     Fine-tuning: {}   '
        '========================='.format(phase_message))
  checkpoint_path = '{}/checkpoints/{}_{}_checkpoint.pt'.format(
      __args__.workdir, checkpoint_prefix, hex(__args__.br_pc))
  load_checkpoint_or_train_and_create_checkpoint(
      model_wrapper, checkpoint_path,
      __args__.fine_tuning_training_steps, 0.0,
      fc_regularization_coeff)

def find_most_useful_filters(model_wrapper):
  print('Finding the most useful filters in the validation set.')
  num_pruned_filters_per_slice = model_wrapper.model.config['pruned_conv_filters']

  loss_values_per_slice = model_wrapper.model.group_lasso_loss_values()
  useful_filters = []
  for loss_values, num_pruned_filters in zip(
      loss_values_per_slice,
      num_pruned_filters_per_slice):
    top_channels = torch.topk(
        loss_values,
        num_pruned_filters).indices
    useful_filters.append(sorted(list(top_channels.cpu().numpy())))
  return useful_filters


def full_evaluation(model_wrapper, include_training_set=False):
  print('Evaluating the model on all traces')
  if include_training_set:
    traces = __args__.evaluation_traces + __args__.training_traces
  else:
    traces = __args__.evaluation_traces
  for trace in traces:
    corrects, total, _ = model_wrapper.eval([trace])
    accuracy = 0 if total == 0 else corrects/total
    print('accuracy of {}: {} out of {} ({}%)'.format(
        trace, corrects, total, accuracy*100.0))
    with open('{}/results_evaluation'.format(__args__.workdir), 
              'a') as results_log:
      trace_name = os.path.splitext(os.path.basename(trace))[0]
      print('{}_{},{},{},{}'.format(
          trace_name, hex(__args__.br_pc), accuracy, corrects, total),
          file=results_log)

def float_training(model_wrapper):
  full_precision_training_phase(model_wrapper)
  return model_wrapper

def tarsa_ternary_training(model_wrapper):
  model_wrapper = change_training_phase_knobs(
      model_wrapper,
      {'quantize_convolution': True,
       'quantize_final_fc': True}
  )
  full_precision_training_phase(model_wrapper)
  full_evaluation(model_wrapper)

  model_wrapper = convert_convolutions_to_luts(model_wrapper)

  fine_tune_the_model(model_wrapper, 'After converting convolutions to LUTs',
                      'final_intel_ternary')

  return model_wrapper

def mini_branchnet_training(model_wrapper):
  model_wrapper = change_training_phase_knobs(
      model_wrapper,
      {'quantize_convolution': True,
       'quantize_sumpooling': True,
       'quantize_hidden_fc': True}
  )
  full_precision_training_phase(model_wrapper)
  #full_evaluation(model_wrapper)

  #model_wrapper = prune_convolution_filters(model_wrapper)
  #fine_tune_the_model(
  #    model_wrapper, 'After pruning conv filters', 'pruned_convs')
  #full_evaluation(model_wrapper)

  model_wrapper = convert_convolutions_to_luts(model_wrapper)
  fine_tune_the_model(model_wrapper, 'After converting convolutions to LUTs',
                      'lut_conv')
  #full_evaluation(model_wrapper)

  model_wrapper = change_training_phase_knobs(
      model_wrapper,
      {'freeze_sumpooling_batchnorm_params': True,}
  )
  fine_tune_the_model(
      model_wrapper,
      'After freezing sumpooling normalization, '
      'while regularizing fc weights',
      'frozen_sumpooling',
      fc_regularization_coeff=__args__.fc_regularization_coeff)
  #full_evaluation(model_wrapper)

  model_wrapper = change_training_phase_knobs(
      model_wrapper,
      {'prune_fc_layers': True,}
  )
  model_wrapper.model.setup_fc_pruning_masks()
  fine_tune_the_model(model_wrapper,
                      'After pruning the fully-connected layers',
                      'pruned_fc_layers')
  #full_evaluation(model_wrapper)

  model_wrapper = change_training_phase_knobs(
      model_wrapper,
      {'freeze_hidden_fc_params': True,}
  )
  fine_tune_the_model(model_wrapper,
                      'After freezing hidden fully-connected layers',
                      'final')
  return model_wrapper

def main():
  """Main function."""
  params = vars(__args__)
  for key in sorted(params):
    print('{}: {}'.format(key, params[key]))

  os.makedirs(__args__.workdir, exist_ok=True)
  os.makedirs(__args__.workdir+'/checkpoints', exist_ok=True)
  
  with open(__args__.config_file, 'r') as f:
    config = yaml.safe_load(f)

  model_wrapper = create_full_precision_branchnet_model(config)

  training_functions = {
    'float': float_training,
    'mini': mini_branchnet_training,
    'tarsa': tarsa_ternary_training,
  }
  model_wrapper = training_functions[__args__.training_mode](model_wrapper)

  full_evaluation(model_wrapper, include_training_set=False)



if __name__ == '__main__':
  main()
