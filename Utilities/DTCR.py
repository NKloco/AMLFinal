"""
This module will provide the functionality required to recreate the results of
the deep temporal clustering representation(DTCR) algorithm.
"""
import math
import random
import torch
import torch.nn as nn
from Utilities.DRNN import BidirectionalDRNN

FAKE_SAMPLE_ALPHA = 0.2  # As set in the article


class DTCRConfig(object):
    """
    This class shall hold the information required by a DTCRModel.
    """
    batch_size = None
    input_size = None
    hidden_size = [100, 50, 50]
    dilations = [1, 4, 16]
    num_steps = None
    embedding_size = None
    learning_rate = 5e-3
    encoder_cell_type = 'GRU'
    decoder_cell_type = 'GRU'
    coefficient_lambda = 1
    class_num = None
    de_noising = True
    sample_loss = True


class DTCRModel(nn.Module):
    def __init__(self, config):
        super(DTCRModel, self).__init__()

        self._config = config
        self._encoder = self._generate_encoder()
        self._decoder = self._generate_decoder()
        self._classifier = self._generate_classifier()

    def _generate_encoder(self):
        """
        Generates the encoder of the DTCR model.
        The decoder is a bi-directional dilated RNN.
        """
        encoder = \
            BidirectionalDRNN(self._config.input_size,
                              self._config.hidden_size,
                              len(self._config.hidden_size),
                              cell_type=self._config.encoder_cell_type,
                              batch_first=True,
                              dilations=self._config.dilations)

        return encoder

    @property
    def encoder(self):
        return self._encoder

    def _generate_decoder(self):
        decoder = DTCRDecoder(self._config)
        return decoder

    @property
    def decoder(self):
        return self._decoder

    def _generate_classifier(self):
        pass

    def get_latent_representation(self, hidden_output):
        latent_space_last_hidden_outputs = []
        for layer_output_tensor in hidden_output:
            # The inputs are padded so that there's exactly enough
            # time steps for a full dilation, since that padding
            # happens, we need to take the modulo of the steps from
            # the hidden outputs.
            last_hidden = layer_output_tensor.select(
                0, self._config.num_steps % layer_output_tensor.shape[0])

            latent_space_last_hidden_outputs.append(last_hidden)

        # There are 6 layers (3 layers for each direction), which we
        # combine for the latent representation
        return torch.cat(latent_space_last_hidden_outputs, dim=1)

    def forward(self, inputs):
        # inputs of shape (Batch, Time Steps, Single step size)

        _, hidden_outputs = self.encoder(inputs)
        # hidden_outputs: list of length of layers * directions (6)
        # each item of shape (dilation, batch, hidden size)

        latent_repr = self.get_latent_representation(hidden_outputs)
        # latent_repr: (batch, latent_space_size)

        prep_for_decoder = latent_repr.repeat(1, 1, 1).transpose(0, 1)
        # prep_for_decoder: (batch, time steps [1], input size)

        reconstructed_inputs = self.decoder(prep_for_decoder)
        # reconstructed_inputs have the same shape as the input

        return inputs, latent_repr, reconstructed_inputs

    def train(self, inputs):
        # For now the I skip the classifier's loss
        # fake_samples = create_fake_time_series(inputs)

        _, hidden_outputs = self.encoder.forward(inputs)
        latent_representation = self.get_latent_representation(hidden_outputs)

        reconstructed_inputs = self._decoder.forward(latent_representation)
        return latent_representation


class DTCRDecoder(nn.Module):
    def __init__(self, config):
        super(DTCRDecoder, self).__init__()
        self._config = config

        if self._config.decoder_cell_type == "GRU":
            cell = nn.GRU
        elif self._config.decoder_cell_type == "RNN":
            cell = nn.RNN
        elif self._config.decoder_cell_type == "LSTM":
            cell = nn.LSTM
        else:
            raise NotImplementedError

        self._number_of_units = sum(self._config.hidden_size) * 2

        self._rnn = cell(self._number_of_units, self._number_of_units,
                         dropout=0, batch_first=True)

        self._linear = nn.Linear(self._number_of_units,
                                 self._config.input_size)

    def forward(self, inputs, hidden=None, predict_length=None):
        if predict_length is None:
            predict_length = self._config.num_steps

        if hidden is None:
            batch_size = inputs.shape[0]

            # Hidden is of (layers, batch_size, hidden_units)
            hidden = torch.zeros(1, batch_size, self._number_of_units)

        series_prediction = []
        rnn_out = inputs
        for _ in range(predict_length):
            rnn_out, hidden = self._rnn(rnn_out, hidden)
            step_prediction = self._linear(rnn_out)

            series_prediction.append(step_prediction)

        predicted_series = torch.cat(series_prediction, dim=1)
        return predicted_series


def create_fake_time_series(sample, fake_alpha=FAKE_SAMPLE_ALPHA):
    fake_sample = list(sample)

    # The number of samples from the series to shuffle
    time_steps_to_shuffle = math.floor(len(sample) * fake_alpha)

    # The indices to shuffle
    indices_to_shuffle = random.sample(range(len(sample)),
                                       time_steps_to_shuffle)

    while len(indices_to_shuffle) > 1:
        # Choosing the indices to shuffle (The last one with a random one)
        last_index = indices_to_shuffle.pop()  # Decrease the length by 1
        random_index_to_swap = indices_to_shuffle[
            random.randint(0, len(indices_to_shuffle) - 1)]

        # Swapping the items
        swap_temp = fake_sample[last_index]
        fake_sample[last_index] = fake_sample[random_index_to_swap]
        fake_sample[random_index_to_swap] = swap_temp

    return fake_sample


def main():
    print("Testing the DTCR functionality...")
    # Creating a sample with 1000 values to check the fake
    # sampling functionality.
    sample = range(1000)
    fake_sample = create_fake_time_series(sample)

    shuffled_count = 0

    for index in range(len(sample)):
        if sample[index] != fake_sample[index]:
            shuffled_count += 1

    print("For a sample of length {}, there were {} shuffles".format(
        len(sample), shuffled_count))
    print("Which is {} of the series.".format(shuffled_count/len(sample)))

    print("DTCR functionality finished testing.")


if __name__ == "__main__":
    main()
