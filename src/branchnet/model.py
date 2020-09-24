"""
Definition of a BranchNet model in Pytorch
"""

import copy
import math
import numpy as np
import random
import torch
import torch.nn as nn


class BranchNetTrainingPhaseKnobs:
  """Set of knobs for reconfiguring a BranchNet model during training phases.
  """
  def __init__(self):
    self.quantize_convolution = False
    self.quantize_sumpooling = False
    self.quantize_hidden_fc = False
    self.quantize_final_fc = False

    self.prune_filters = False
    self.prune_fc_layers = False

    self.lut_convolution = False

    self.freeze_sumpooling_batchnorm_params = False
    self.freeze_hidden_fc_params = False


class Quantize(torch.autograd.Function):
  """Quantizes its input within [0,1] or [-1,+1]"""

  @staticmethod
  def forward(ctx, x, unsigned, precision):
    if unsigned:
      scale = ((1 << precision) - 1)
      return torch.round(x * scale) / scale
    else:
      if precision == 1:
        return -1 + 2 * (x > 0).float()
      else: 
        scale = ((1 << (precision - 1)) - 1)
        return torch.round(x * scale) / scale

  @staticmethod
  def backward(ctx, grad_output):
    return grad_output.clone(), None, None


def lists_have_equal_length(list_of_lists):
  """helper function to check that the length of lists are equals"""
  set_of_lengths = set(map(len, list_of_lists))
  return len(set_of_lengths) <= 1


def extract_slice_history(x, config, global_shift, slice_id):
  """Extract a portion of history for a slice."""

  total_history_size = x.shape[1]
  slice_size = config['history_lengths'][slice_id]
  pooling_width = config['pooling_widths'][slice_id]
  assert slice_size <= total_history_size

  if config['shifting_pooling'][slice_id]:
    slice_shift = global_shift % pooling_width
    inputs = []
    for i in range(x.shape[0]):
      slice_end = total_history_size - slice_shift[i]
      slice_start = slice_end - slice_size
      inputs.append(x[i, slice_start:slice_end])
    return torch.stack(inputs)
  else:
    return x[:, -slice_size:]


class Slice(nn.Module):
  """A Pytorch neural network module class to define a BranchNet slice
    corresponding to some portion of the history.
  """

  def __init__(self, config, slice_id, training_phase_knobs):
    """Creates all the layers and computes the expected output size.
    """
    super(Slice, self).__init__()
    history_length = config['history_lengths'][slice_id]
    conv_filters = config['conv_filters'][slice_id]
    conv_width = config['conv_widths'][slice_id]
    pooling_width = config['pooling_widths'][slice_id]
    embedding_dims = config['embedding_dims']
    pc_hash_bits = config['pc_hash_bits']
    hash_dir_with_pc = config['hash_dir_with_pc']

    # remember slice configuration
    self.config = config
    self.slice_id = slice_id
    self.lut_convolution = training_phase_knobs.lut_convolution
    self.quantize_sumpooling = training_phase_knobs.quantize_sumpooling
    self.training_phase_knobs = training_phase_knobs

    if training_phase_knobs.prune_filters:
      self.pruning_mask = nn.Parameter(torch.zeros(
          1, self.config['conv_filters'][slice_id], 1), requires_grad=False)

    # Declare all the neural network layers
    index_width = pc_hash_bits if hash_dir_with_pc else (pc_hash_bits + 1)
    if self.lut_convolution:
      if config['combined_hash_convolution']:
        assert not hash_dir_with_pc
        self.build_hashing_metadata()
        self.combined_lookup_table = nn.Embedding(
            2 ** config['combined_hash_convolution_width'],
            conv_filters)
        self.combined_lookup_table.weight.requires_grad = False
      else:
        self.lookup_tables = nn.ModuleList()
        for i in range(conv_width):
          self.lookup_tables.append(
              nn.Embedding(2 ** index_width, conv_filters))
          self.lookup_tables[i].weight.requires_grad = False 
    else:
      if config['combined_hash_convolution']:
        assert not hash_dir_with_pc
        self.build_hashing_metadata()
        self.combined_embedding_table = nn.Embedding(
            2 ** config['combined_hash_convolution_width'],
            embedding_dims)
        self.combined_conv = nn.Conv1d(embedding_dims, conv_filters, 1)
        self.batchnorm = nn.BatchNorm1d(conv_filters)
      else:
        self.embedding_table = nn.Embedding(2 ** index_width, embedding_dims)
        self.conv = nn.Conv1d(embedding_dims, conv_filters, conv_width)
        self.batchnorm = nn.BatchNorm1d(conv_filters)

    self.pooling = nn.AvgPool1d(pooling_width, padding=0)

    if self.quantize_sumpooling:
      self.pooling_batchnorm = nn.BatchNorm1d(
          conv_filters * self.config['sumpooling_copies'])
      if training_phase_knobs.freeze_sumpooling_batchnorm_params:
        self.pooling_batchnorm.bias.requires_grad = False
        self.pooling_batchnorm.weight.requires_grad = False
    else:
      self.pooling_batchnorm = nn.BatchNorm1d(conv_filters)


    # compute the slice output size
    if pooling_width == -1 or (config['shifting_pooling'][slice_id]
                               and config['sum_all_if_shifting_pool']):
      pooling_output_size = 1
    elif pooling_width > 0: 
      conv_output_size = (history_length - conv_width + 1)
      pooling_output_size = conv_output_size // pooling_width
    else:
      pooling_output_size = (history_length - conv_width + 1)
    self.total_output_size = pooling_output_size * conv_filters
    if self.quantize_sumpooling:
      self.total_output_size *= self.config['sumpooling_copies']


  def build_hashing_metadata(self):
    num_input_bits = ((self.config['pc_hash_bits'] + 1) *
                      max(self.config['conv_widths']))
    num_output_bits = self.config['combined_hash_convolution_width']

    assert num_output_bits < 32
    self.hash_metadata = nn.Parameter(torch.randint(
        0, 2 ** num_output_bits, size=[num_input_bits], dtype=torch.int64), requires_grad=False)

  def hash_using_metadata(self, x, conv_width):
    batch_size = x.shape[0]
    available_history = x.shape[1]
    output_history = available_history + 1 - conv_width
    bits_per_conv_pos = self.config['pc_hash_bits'] + 1
    zero_tensor = torch.zeros(1,
                      dtype=torch.int64, device=x.device)
    out = torch.zeros(batch_size, output_history,
                      dtype=torch.int64, device=x.device)

    for conv_pos in range(conv_width):
      history_slice = x[:, available_history - conv_pos - output_history: available_history - conv_pos]
      for bit in range(bits_per_conv_pos):
        metadata_idx = conv_pos * bits_per_conv_pos + bit
        xor_pattern = self.hash_metadata[metadata_idx: metadata_idx + 1]
        out = out ^ torch.where((history_slice >> bit) & 1 == 1, xor_pattern, zero_tensor)
    
    return out


  def forward(self, x):
    history_length = self.config['history_lengths'][self.slice_id]
    conv_filters = self.config['conv_filters'][self.slice_id]
    conv_width = self.config['conv_widths'][self.slice_id]
    pooling_width = self.config['pooling_widths'][self.slice_id]

    if self.lut_convolution:
      if self.config['combined_hash_convolution']:
        x = self.hash_using_metadata(x, conv_width)
        x = self.combined_lookup_table(x)
        x = x.transpose(1, 2).contiguous()
        if self.training_phase_knobs.prune_filters:
          x = x * self.pruning_mask
      else:
        batch_size = x.shape[0]
        num_channels = conv_filters
        conv_outputs = torch.zeros(batch_size,
                                   num_channels,
                                   history_length - conv_width + 1,
                                   device=x.device)

        for conv_pos in range(conv_width):
          temp = self.lookup_tables[conv_pos](
              x[:,conv_pos:history_length-conv_width+conv_pos+1])
          temp = temp.transpose(1, 2)
          conv_outputs += temp
        x = conv_outputs
        x = self.convolution_activation(x)
    else:
      # convolution and batch norm layers
      if self.config['combined_hash_convolution']:
        x = self.hash_using_metadata(x, conv_width)
        x = self.combined_embedding_table(x)
        x = torch.transpose(x, 1, 2)
        x = self.combined_conv(x)
        x = self.batchnorm(x)
        x = self.convolution_activation(x)
      else:
        x = self.embedding_table(x)
        x = torch.transpose(x, 1, 2)
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.convolution_activation(x)

    # pooling
    if pooling_width == -1 or (self.config['shifting_pooling'][self.slice_id]
                               and self.config['sum_all_if_shifting_pool']):
      x = torch.sum(x, 2, keepdim=True)
    elif pooling_width > 0:
      x = self.pooling(x) * pooling_width

    x = self.sumpooling_activation(x)

    return x.view(-1, self.total_output_size)

  def get_output_size(self):
    """Returns the expected output size for the slice
    """
    return self.total_output_size

  def convolution_activation(self, x):
    """Returns post- and pre- quantization activations."""
    relu_act = nn.ReLU(inplace=True)
    sigmoid_act = nn.Sigmoid()
    tanh_act = nn.Tanh()
    quantize = Quantize.apply

    conv_activation_type = self.config['conv_activation']
    conv_quantization_bits = self.config['conv_quantization_bits']

    if conv_activation_type == 'relu':
      x = relu_act(x)
      if self.training_phase_knobs.prune_filters:
        x = x * self.pruning_mask
      assert not self.training_phase_knobs.quantize_convolution
      return x
    if conv_activation_type == 'sigmoid':
      x = sigmoid_act(x)
      if self.training_phase_knobs.prune_filters:
        x = x * self.pruning_mask
      if self.training_phase_knobs.quantize_convolution:
        assert conv_quantization_bits > 0
        return quantize(x, True, conv_quantization_bits)
      else:
        return x
    if conv_activation_type == 'tanh':
      x = tanh_act(x)
      if self.training_phase_knobs.prune_filters:
        x = x * self.pruning_mask
      if self.training_phase_knobs.quantize_convolution:
        assert conv_quantization_bits > 0
        return quantize(x, False, conv_quantization_bits)
      else:
        return x

    assert False

  def sumpooling_activation(self, x):
    if self.quantize_sumpooling:
      repeat_pattern = [1] * len(x.shape)
      repeat_pattern[1] = self.config['sumpooling_copies']
      x = x.repeat(*repeat_pattern)

    tanh_act = nn.Tanh()
    hardtanh_act = nn.Hardtanh()
    sigmoid_act = nn.Tanh()
    hardsigmoid_act = nn.Hardtanh(min_val=0.0, max_val=1.0)
    quantize = Quantize.apply

    activation = self.config['sumpooling_activation']
    quantization_bits = self.config['sumpooling_quantization_bits']

    if activation == 'none':
      assert quantization_bits == 0
      return x
    if activation == 'bn_only':
      assert quantization_bits == 0
      return self.pooling_batchnorm(x)
    if activation == 'tanh':
      x = tanh_act(self.pooling_batchnorm(x))
      if not self.quantize_sumpooling or quantization_bits == 0:
        return x
      else:
        return quantize(x, False, quantization_bits)
    if activation == 'hardtanh':
      x = hardtanh_act(self.pooling_batchnorm(x))
      if not self.quantize_sumpooling or quantization_bits == 0:
        return x
      else:
        return quantize(x, False, quantization_bits)
    if activation == 'sigmoid':
      x = sigmoid_act(self.pooling_batchnorm(x))
      if not self.quantize_sumpooling or quantization_bits == 0:
        return x
      else:
        return quantize(x, True, quantization_bits)
    if activation == 'hardsigmoid':
      x = hardsigmoid_act(self.pooling_batchnorm(x))
      if not self.quantize_sumpooling or quantization_bits == 0:
        return x
      else:
        return quantize(x, True, quantization_bits)

    assert False

  def setup_pruning_mask(self, useful_channels_for_slice):
    indices = torch.LongTensor(useful_channels_for_slice).to(
        self.pruning_mask.device).unsqueeze(0).unsqueeze(2)
    self.pruning_mask.scatter_(1, indices, 1)
    print(self.pruning_mask.view(-1))


class FCLayer(nn.Module):
  def __init__(self, input_dim, output_dim, *, activation, quantize,
               quantized_act_bits, quantized_weight_bits, freeze_params,
               use_pruning_mask):
    super(FCLayer, self).__init__()
    self.use_pruning_mask = use_pruning_mask
    self.activation = activation
    self.quantize = quantize
    self.quantized_act_bits = quantized_act_bits
    self.quantized_weight_bits = quantized_weight_bits

    self.weight = nn.Parameter(torch.empty(output_dim, input_dim),
                                     requires_grad=not freeze_params)
    self.bias = nn.Parameter(torch.empty(output_dim),
                                   requires_grad=not freeze_params)
    if activation is not None:
      self.batchnorm = nn.BatchNorm1d(output_dim)

    if use_pruning_mask:
      self.pruning_mask = nn.Parameter(torch.zeros_like(self.bias),
                                    requires_grad=False)

    self.randomize_weights()

  def forward(self, x):
    quantize = Quantize.apply

    if self.quantize and self.quantized_weight_bits > 0: 
      self.state_dict()['weight'][:] = torch.clamp(
          self.weight.data, -1, 1)
      weight = quantize(self.weight, False, self.quantized_weight_bits)
    else:
      weight = self.weight

    x = nn.functional.linear(x, weight, bias=self.bias)
    if self.activation is not None:
      x = self.activation_layer(x)
    if self.use_pruning_mask:
      x = x * self.pruning_mask
    return x

  def activation_layer(self, x):
    x = self.batchnorm(x)

    quantize = Quantize.apply
    relu_act = nn.ReLU(inplace=True)
    sigmoid_act = nn.Sigmoid()
    tanh_act = nn.Tanh()
    hardtanh_act = nn.Hardtanh()
    quantize_act = self.quantize and self.quantized_act_bits > 0

    if self.activation == 'relu':
      assert not quantize_act
      x = relu_act(x)
    elif self.activation == 'sigmoid':
      x = sigmoid_act(x)
      if quantize_act:
        x = quantize(x, True, self.quantized_act_bits)
    elif self.activation == 'tanh':
      x = tanh_act(x)
      if quantize_act:
        x = quantize(x, False, self.quantized_act_bits)
    elif self.activation == 'hardtanh':
      x = hardtanh_act(x)
      if quantize_act:
        x = quantize(x, False, self.quantized_act_bits)
    else:
      assert False

    return x

  def randomize_weights(self):
    output_dim = self.weight.shape[0]
    input_dim = self.weight.shape[1]
    glorot_init_bound = math.sqrt(2. / (input_dim + output_dim))
    if self.quantize and self.quantized_weight_bits > 0:
      self.weight.data.uniform_(-1, 1)
    else:
      self.weight.data.uniform_(-glorot_init_bound, +glorot_init_bound)
    self.bias.data.uniform_(-glorot_init_bound, +glorot_init_bound)

  def l1_loss(self):
    return torch.sum(torch.abs(self.weight))

  def setup_pruning_mask(self, top_neuron_indices):
    self.pruning_mask.scatter_(0, top_neuron_indices, 1)
    print(self.pruning_mask)


class BranchNetMLP(nn.Module):
  def __init__(self, config, training_phase_knobs, flattened_input_dim):
    super(BranchNetMLP, self).__init__()
    self.config = config
    self.training_phase_knobs = training_phase_knobs
    self.hidden_layers = nn.ModuleList()

    next_input_dim = flattened_input_dim
    for hidden_output_dim in self.config['hidden_neurons']:
      assert hidden_output_dim > 0
      self.hidden_layers.append(FCLayer(
          next_input_dim, hidden_output_dim,
          activation=self.config['hidden_fc_activation'],
          quantize=training_phase_knobs.quantize_hidden_fc,
          quantized_act_bits=self.config['hidden_fc_activation_quantization_bits'],
          quantized_weight_bits=self.config['hidden_fc_weight_quantization_bits'],
          freeze_params=training_phase_knobs.freeze_hidden_fc_params,
          use_pruning_mask=training_phase_knobs.prune_fc_layers))
      next_input_dim = hidden_output_dim

    self.last_layer = FCLayer(
        next_input_dim, 1,
        activation=None,
        quantize=training_phase_knobs.quantize_final_fc,
        quantized_act_bits=0,
        quantized_weight_bits=self.config['final_fc_weight_quantization_bits'],
        freeze_params=False,
        use_pruning_mask=False)

  def forward(self, x):
    for i in range(len(self.hidden_layers)):
      x = self.hidden_layers[i](x)
    x = self.last_layer(x)
    return x.squeeze(dim=1)

  def randomize_weights(self):
    for i in range(len(self.config['hidden_neurons'])):
      self.hidden_layers[i].randomize_weights()
    self.last_layer.randomize_weights()

  def l1_loss(self):
    loss = self.last_layer.l1_loss()
    # Skip the first hidden fc for regularization.
    for i in range(1, len(self.config['hidden_neurons'])):
      loss += self.hidden_layers[i].l1_loss()
    return loss

  def setup_fc_pruning_masks(self):
    for i in range(0, len(self.config['hidden_neurons'])):
      if i == len(self.config['hidden_neurons']) - 1:
        next_layer = self.last_layer
      else:
        next_layer = self.hidden_layers[i + 1]
      sum_next_weights = torch.sum(torch.abs(next_layer.weight), dim=[0])
      top_neuron_indices = torch.topk(
          sum_next_weights,
          self.config['pruned_hidden_neurons'][i]).indices
      self.hidden_layers[i].setup_pruning_mask(top_neuron_indices)

class BranchNet(nn.Module):
  """
  A Pytorch neural network module class to define BranchNet architecture.
  """

  def __init__(self, config, training_phase_knobs):
    super(BranchNet, self).__init__()

    assert lists_have_equal_length(
        [config['history_lengths'], config['conv_filters'],
         config['conv_widths'], config['pooling_widths']])

    self.history_lengths = config['history_lengths']
    self.config = config
    self.linear_pruning_mask = None
    self.quantize_fc = False
    self.training_phase_knobs = training_phase_knobs

    num_slices = len(self.history_lengths)
    self.slices = nn.ModuleList()
    concatenated_slices_output_size = 0
    for slice_id in range(num_slices):
      if config['conv_filters'][slice_id] > 0:
        self.slices.append(Slice(config, slice_id,  training_phase_knobs))
        concatenated_slices_output_size += self.slices[slice_id].get_output_size()
      else:
        self.slices.append(nn.ReLU()) #insert dummy module instead of a slice

    self.mlp = BranchNetMLP(self.config, self.training_phase_knobs,
                            concatenated_slices_output_size)

  def forward(self, x):
    #pylint: disable=arguments-differ
    #It is expected to change forward() arguments.
    #if self.linear_pruning_mask is not None:
    #  self.state_dict()['linear.weight'][:] = self.linear.weight * self.linear_pruning_mask

    if any(self.config['shifting_pooling']):
      global_shift = np.random.randint(max(self.config['pooling_widths']), size=(x.shape[0]))
    else:
      global_shift = None

    slice_outs = []
    num_slices = len(self.history_lengths)

    for slice_id in range(num_slices):
      if self.config['conv_filters'][slice_id] > 0:
        x_ = extract_slice_history(x, self.config, global_shift, slice_id)
        x_ = self.slices[slice_id](x_)
        slice_outs.append(x_)

    x = torch.cat(slice_outs, dim=1)
    x = self.mlp(x)
    return x

  def train(self, mode=True):
    super(BranchNet, self).train(mode)
    if self.training_phase_knobs.freeze_sumpooling_batchnorm_params:
      for slice_id in range(len(self.history_lengths)):
        self.slices[slice_id].pooling_batchnorm.eval()

  def reinitialize_fc_weights(self):
    self.mlp.randomize_weights()

  def prune_hidden_fc(self, n):
    self.linear_pruning_mask = self.linear.weight.clone().detach().zero_()
    selected_indices = torch.topk(abs(self.linear.weight), n, dim=1)
    self.linear_pruning_mask.scatter_(1, selected_indices[1], 1)
    

  def linear_regularization_loss(self):
    return torch.norm(self.linear.weight)

  def group_lasso_loss_values(self):
    """ Get the loss term for convolution filters group lassos
    """
    lasso_groups = []
    num_slices = len(self.history_lengths)

    for slice_id in range(num_slices):
      conv_weights_squared = self.slices[slice_id].conv.weight.pow(2)
      lasso_groups.append(torch.sqrt(conv_weights_squared.sum(dim=[1, 2])))

    return lasso_groups

  def group_lasso_loss(self):
    """ Get the loss term for convolution filters group lassos
    """
    lasso_groups = []
    #linear_weights_squared = self.linear.weight.pow(2)
    if len(self.mlp.hidden_layers) > 0:
      linear_weights_squared = self.mlp.hidden_layers[0].weight.pow(2)
    else:
      linear_weights_squared = self.mlp.last_layer.weight.pow(2)
    num_slices = len(self.history_lengths)

    i = 0
    for slice_id in range(num_slices):
      # Grouping Convolution Weights.
      conv_weights_squared = self.slices[slice_id].conv.weight.pow(2)
      lasso_groups.append(conv_weights_squared.sum(dim=[1, 2]))
      lasso_groups.append(self.slices[slice_id].embedding_table.weight.pow(2).sum(dim=[1]))

      # Grouping Fully-connected Weights.
      slice_output_size = self.slices[slice_id].get_output_size()
      num_filters = self.config['conv_filters'][slice_id]

      slice_linear_weights_squared = (
          linear_weights_squared[:, i:i+slice_output_size])
      i += slice_output_size

      slice_linear_weights_squared = slice_linear_weights_squared.view(
          -1, num_filters, slice_output_size // num_filters)
      lasso_groups.append(slice_linear_weights_squared.sum(dim=[0, 2]))

    return torch.sum(torch.sqrt(torch.cat(lasso_groups, dim=0)))

  def fc_weights_l1_loss(self):
    return self.mlp.l1_loss()

  def copy_from_other_model(self, other_model):
    for key in self.state_dict():
      if key in other_model.state_dict():
        if len(other_model.state_dict()[key].shape) > 0:
          if (self.state_dict()[key].shape
              == other_model.state_dict()[key].shape):
            self.state_dict()[key][:] = other_model.state_dict()[key]
          else:
            print('Warning: did not copy', key)
    self.copy_masks(other_model)

  def setup_fc_pruning_masks(self):
    self.mlp.setup_fc_pruning_masks()

  def quantize_luts(self, x, conv_activation_type, conv_quantization_bits):
    quantize = Quantize.apply
    if conv_activation_type == 'relu':
      act = nn.ReLU()
      assert conv_quantization_bits == 0
      x = act(x)
    elif conv_activation_type == 'sigmoid':
      act = nn.Sigmoid()
      x = act(x)
      if conv_quantization_bits > 0:
        x = quantize(x, True, conv_quantization_bits)
    elif conv_activation_type == 'tanh':
      act = nn.Tanh()
      x = act(x)
      if conv_quantization_bits > 0:
        x = quantize(x, False, conv_quantization_bits)
    return x

  def load_convolution_luts(self, trained_branchnet):
    assert self.training_phase_knobs.lut_convolution is True
    assert trained_branchnet.training_phase_knobs.lut_convolution is False

    conv_filters = self.config['conv_filters']
    conv_widths = self.config['conv_widths']
    conv_activation_type = self.config['conv_activation']
    conv_quantization_bits = self.config['conv_quantization_bits']
    self.copy_from_other_model(trained_branchnet)

    if self.config['combined_hash_convolution']:
      for slice_id in range(len(conv_filters)):
        orig_embedding = trained_branchnet.state_dict()[
            'slices.{}.combined_embedding_table.weight'.format(slice_id)]
        conv_weight = trained_branchnet.state_dict()[
            'slices.{}.combined_conv.weight'.format(slice_id)]
        conv_bias = trained_branchnet.state_dict()[
            'slices.{}.combined_conv.bias'.format(slice_id)]
        batchnorm_weight = trained_branchnet.state_dict()[
            'slices.{}.batchnorm.weight'.format(slice_id)]
        batchnorm_bias = trained_branchnet.state_dict()[
            'slices.{}.batchnorm.bias'.format(slice_id)]
        batchnorm_mean = trained_branchnet.state_dict()[
            'slices.{}.batchnorm.running_mean'.format(slice_id)]
        batchnorm_var = trained_branchnet.state_dict()[
            'slices.{}.batchnorm.running_var'.format(slice_id)]

        new_embedding = self.state_dict()[
            'slices.{}.combined_lookup_table.weight'.format(slice_id)]

        conv_weights_transposed = conv_weight[:, :, 0].transpose(0, 1)
        new_embedding[:] = torch.matmul(orig_embedding, conv_weights_transposed)
        new_embedding += conv_bias.view(1, conv_filters[slice_id])
        new_embedding -= batchnorm_mean.view(1, conv_filters[slice_id])
        new_embedding /= torch.sqrt(
            batchnorm_var.view(1, conv_filters[slice_id]) + 1e-5)
        new_embedding *= batchnorm_weight.view(1, conv_filters[slice_id])
        new_embedding += batchnorm_bias.view(1, conv_filters[slice_id])

        new_embedding[:] = self.quantize_luts(
            new_embedding, conv_activation_type, conv_quantization_bits)
        final_luts = new_embedding

        if conv_quantization_bits > 0:
          max_val = 1
          if conv_activation_type in ['sigmoid' or 'cross_channel_sigmoid_binarize']:
            min_val = 0
            num_slices = (2 ** conv_quantization_bits - 1)
          elif conv_activation_type == 'tanh':
            min_val = -1
            num_slices = max(1, 2 ** conv_quantization_bits - 2)
          else:
            assert False

          step = (max_val - min_val) / num_slices
          bin_boundaries = np.arange(min_val, max_val, step) + (step/2)
          bin_values = np.concatenate([np.arange(min_val, max_val, step), np.array([max_val])])
          digitized_filters = np.digitize(final_luts.cpu().numpy(), bin_boundaries)
          for bin_id in range(len(bin_boundaries) + 1):
            print('Number of {}: {}'.format(bin_values[bin_id], np.sum(digitized_filters == bin_id)))
        
    else:
      for slice_id in range(len(conv_filters)):
        orig_embedding = trained_branchnet.state_dict()[
            'slices.{}.embedding_table.weight'.format(slice_id)]
        conv_weight = trained_branchnet.state_dict()[
            'slices.{}.conv.weight'.format(slice_id)]
        conv_bias = trained_branchnet.state_dict()[
            'slices.{}.conv.bias'.format(slice_id)]
        batchnorm_weight = trained_branchnet.state_dict()[
            'slices.{}.batchnorm.weight'.format(slice_id)]
        batchnorm_bias = trained_branchnet.state_dict()[
            'slices.{}.batchnorm.bias'.format(slice_id)]
        batchnorm_mean = trained_branchnet.state_dict()[
            'slices.{}.batchnorm.running_mean'.format(slice_id)]
        batchnorm_var = trained_branchnet.state_dict()[
            'slices.{}.batchnorm.running_var'.format(slice_id)]
        
        list_new_embeddings = []
        for conv_pos in range(conv_widths[slice_id]):
          new_embedding = self.state_dict()[
              'slices.{}.lookup_tables.{}.weight'.format(slice_id, conv_pos)]
          conv_pos_weights = conv_weight[ :, :, conv_pos]
          conv_pos_weights = conv_pos_weights.transpose(0, 1)
          new_embedding[:] = torch.matmul(orig_embedding, conv_pos_weights)
          new_embedding[:] = new_embedding + (
              conv_bias.view(1, conv_filters[slice_id]) / conv_widths[slice_id])
          new_embedding[:] = new_embedding - (
              batchnorm_mean.view(1, conv_filters[slice_id]) / conv_widths[slice_id])
          new_embedding[:] = new_embedding * batchnorm_weight.view(1, conv_filters[slice_id])
          new_embedding[:] = new_embedding / torch.sqrt(
              batchnorm_var.view(1, conv_filters[slice_id]) + 1e-5)
          new_embedding += batchnorm_bias.view(
              1, conv_filters[slice_id]) / conv_widths[slice_id]
          list_new_embeddings.append(new_embedding)

  def setup_conv_pruning_masks(self, useful_channels):
    print('Useful Channels:', useful_channels)
    for slice_id, useful_channels_for_slice in enumerate(useful_channels):
      self.slices[slice_id].setup_pruning_mask(useful_channels_for_slice)
  
  def copy_masks(self, other):
    if (self.training_phase_knobs.prune_filters
        and other.training_phase_knobs.prune_filters):
      num_slices = len(self.config['conv_filters'])
      for slice_id in range(num_slices):
        self.slices[slice_id].pruning_mask = other.slices[slice_id].pruning_mask
