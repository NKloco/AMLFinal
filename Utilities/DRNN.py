"""
Originally taken from
https://github.com/zalandoresearch/pytorch-dilated-rnn/blob/master/drnn.py
Implements a pytorch dilated RNN module.
"""

import torch
from torch.autograd import backward
import torch.nn as nn


class DRNN(nn.Module):
    def __init__(self, n_input, layers, dilations=None, cell_type='GRU',
                 **kwargs):  # kwargs are additional arguments for rnn
        super(DRNN, self).__init__()
        self._cell_type = cell_type
        self._n_layers = layers

        if dilations is None:
            self._dilations = [2 ** i for i in range(len(self._n_layers))]
        elif len(dilations) != len(self._n_layers):
            raise ValueError
        else:
            self._dilations = dilations

        if self._cell_type == "GRU":
            rnn = nn.GRU
        elif self._cell_type == "RNN":
            rnn = nn.RNN
        elif self._cell_type == "LSTM":
            rnn = nn.LSTM
        else:
            raise NotImplementedError

        rnn_layers = []
        for layer_index, n_hidden in enumerate(self._n_layers):
            if layer_index == 0:  # First layer should get input size
                current_layer = rnn(n_input, n_hidden,
                                    **kwargs)
            else:
                current_layer = rnn(self._n_layers[layer_index - 1], n_hidden,
                                    **kwargs)

            rnn_layers.append(current_layer)

        # This is so the model could access the layers parameters, if saved
        # as a list it cannot access them
        self._layers = nn.Sequential(*rnn_layers)

    def forward(self, inputs, hidden=None):
        # inputs: (batch, time steps, input size)
        # hidden: list of number of layers
        # each layer is a list of size of the dilation of the layer
        # and each item is a tensor of size (batch, hidden out)
        batch_size = inputs.shape[0]

        if hidden is None:
            hidden = []
            for n_hidden, dilation in zip(self._n_layers, self._dilations):
                hidden.append([torch.zeros(1, batch_size, n_hidden)]*dilation)

        network_out = []
        network_hidden = []
        rnn_outputs = inputs
        for rnn_layer, dilation, l_hidden in zip(self._layers,
                                                 self._dilations,
                                                 hidden):
            rnn_outputs, rnn_hidden = self._layer_forward(
                rnn_layer, rnn_outputs, l_hidden, dilation)

            network_out.append(rnn_outputs)
            network_hidden.append(rnn_hidden)

        return network_out, network_hidden

    def _layer_forward(self, rnn_layer, inputs, hidden, dilation):
        dilated_inputs = self._split_inputs(inputs, dilation)
        outs = []
        hiddens = []
        for dilation_index in range(dilation):
            current_input = dilated_inputs[dilation_index]
            current_hidden = hidden[dilation_index]
            dilation_out, dilation_hidden = rnn_layer(current_input,
                                                      current_hidden)
            outs.append(dilation_out)
            hiddens.append(dilation_hidden)

        # The hiddens aligned to the beginning of the sequence,
        # according to the dilation we need to rotate it so the last item
        # will represent the last hidden in the layer
        sequence_length = inputs.shape[1]
        rotation = sequence_length % dilation
        hiddens = hiddens[rotation:] + hiddens[:rotation]

        # The outputs are a time series output for each dilation
        # the outputs needs to be ordered again to fit the inputs
        outs = self._merge_outputs(outs, dilation)

        return outs, hiddens

    def _split_inputs(self, inputs, dilation):
        # inputs of shape (batch, time steps, time step size)
        split_inputs = []
        n_time_steps = inputs.shape[1]

        for index in range(dilation):
            indices = [x for x in range(n_time_steps)[index::dilation]]
            indices_tensor = torch.tensor(indices)
            split_inputs.append(torch.index_select(inputs, 1, indices_tensor))

        return split_inputs

    def _merge_outputs(self, outs, dilation):
        # outs: list of tensors of (batch, time steps, time step size)
        total_time_steps = sum([out.shape[1] for out in outs])
        batch_size = outs[0].shape[0]
        time_step_size = outs[0].shape[2]

        out_tensor = torch.zeros(batch_size, total_time_steps, time_step_size)

        for index in range(dilation):
            indices = [x for x in range(total_time_steps)[index::dilation]]
            indices_tensor = torch.tensor(indices)
            out_tensor.index_copy_(1, indices_tensor, outs[index])

        return out_tensor


class BidirectionalDRNN(nn.Module):
    """
    Class that trains 2 DRNNs simultaneously, one regularly and one
    with reversed inputs to make a bidirectional DRNN.
    """
    def __init__(self, n_input, n_hidden, dropout=0,
                 cell_type='GRU', batch_first=False, dilations=None):
        super(BidirectionalDRNN, self).__init__()
        self._number_of_layers = len(n_hidden)

        self._regular_drnn = DRNN(n_input, n_hidden,
                                  dropout=dropout, cell_type=cell_type,
                                  batch_first=batch_first,
                                  dilations=dilations)

        self._backwards_drnn = DRNN(n_input, n_hidden,
                                    dropout=dropout, cell_type=cell_type,
                                    batch_first=batch_first,
                                    dilations=dilations)

    def forward(self, inputs, hidden=None):
        # Previous hidden is a combined hidden layers of the forward and
        # backwards hidden
        if hidden is None:
            regular_hidden = None
            backwards_hidden = None
        else:
            regular_hidden = hidden[:self._number_of_layers]
            backwards_hidden = hidden[self._number_of_layers:]

        regular_outputs, regular_hidden = \
            self._regular_drnn.forward(inputs, regular_hidden)

        reversed_inputs = self._reverse_inputs(inputs)
        backwards_outputs, backwards_hidden = \
            self._backwards_drnn.forward(reversed_inputs, backwards_hidden)

        # We want the outputs to concatenate the same sample's outputs
        # so we need to reverse the outputs of the backwards network
        backwards_outputs_reversed = [
            self._reverse_inputs(back_out) for back_out in backwards_outputs]

        combined_outputs = [
            torch.cat([regular, backward], dim=2) for regular, backward
            in zip(regular_outputs, backwards_outputs_reversed)]

        return combined_outputs, regular_hidden + backwards_hidden

    def _reverse_inputs(self, inputs):
        return torch.flip(inputs, dims=[1])
