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
    # General Settings
    input_size = None  # Size of single step in the time series
    batch_size = None
    num_steps = None  # Length of the time series
    class_num = None  # Number of different labels
    learning_rate = 5e-3
    coefficient_lambda = 1
    de_noising = True
    sample_loss = True
    training_printing_interval = 5
    optimizer = torch.optim.Adam

    # Encoder Settings
    hidden_size = [100, 50, 50]
    dilations = [1, 4, 16]
    encoder_cell_type = 'GRU'

    # Decoder Settings
    decoder_cell_type = 'GRU'
    decoding_criterion = nn.MSELoss

    # Classifier Settings
    classifier_nodes = 128
    classifier_criterion = nn.CrossEntropyLoss


class DTCRModel(nn.Module):
    def __init__(self, config):
        super(DTCRModel, self).__init__()

        self._config = config
        self._latent_space_size = sum(self._config.hidden_size) * 2

        self.encoder = \
            BidirectionalDRNN(self._config.input_size,
                              self._config.hidden_size,
                              cell_type=self._config.encoder_cell_type,
                              batch_first=True,
                              dilations=self._config.dilations)

        self.decoder = DTCRDecoder(self._config)
        self.classifier = classifier = nn.Sequential(
            nn.Linear(self._latent_space_size, self._config.classifier_nodes),
            nn.ReLU(),
            nn.Linear(self._config.classifier_nodes, 2),
            nn.Softmax(dim=1)
        )

    def train_step(self, train_dl, test_dl):
        # Going over the batches
        print_interval = self._config.training_printing_interval
        running_loss = 0.0
        running_recons_loss = 0.0
        running_classify_loss = 0.0

        recons_criterion = self._config.decoding_criterion()
        classify_criterion = self._config.classifier_criterion()
        optimizer = self._config.optimizer(self.parameters(),
                                           eps=self._config.learning_rate)

        for index, (sample_data, sample_label) in enumerate(train_dl):
            optimizer.zero_grad()

            inputs, latent_repr, reconstructed_inputs, classified_outputs\
                = self(sample_data)

            recons_loss = recons_criterion(reconstructed_inputs, inputs)
            classify_loss = classify_criterion(*classified_outputs)

            running_recons_loss += recons_loss.item()
            running_classify_loss += classify_loss.item()
            dtcr_loss = recons_loss + classify_loss
            dtcr_loss.backward()
            optimizer.step()

            running_loss += dtcr_loss.item()
            if index % print_interval == print_interval - 1:
                print('[{}] loss: {}, classify: {}, recons: {}'.format(
                    index + 1, running_loss / print_interval,
                    running_classify_loss / print_interval,
                    running_recons_loss / print_interval))
                running_loss = 0.0
                running_classify_loss = 0.0
                running_recons_loss = 0.0

    def forward(self, inputs):
        # inputs of shape (Batch, Time Steps, Single step size)
        latent_repr, hidden_output = self.encoder_forward(inputs)

        # For decoding we also need a time-step dimension (which is 1)
        # reconstructed_inputs will have the same shape as the input
        latent_repr_for_decoding = latent_repr.repeat(1, 1, 1).transpose(0, 1)
        reconstructed_inputs = self.decoder(latent_repr_for_decoding,
                                            hidden=hidden_output)

        classified_outputs = self.classify_forward(inputs, latent_repr)

        return inputs, latent_repr, reconstructed_inputs, classified_outputs

    def encoder_forward(self, inputs):
        output, hidden_outputs = self.encoder(inputs)

        last_outputs = [out[:, -1, :] for out in output]
        latent_repr = torch.cat(last_outputs, dim=1)

        # hidden_outputs: list of length of layers * directions (6)
        # each is a list of length dilation of the layer
        # and each item is a tensor of (batch, hidden size of layer)
        latent_hidden = self._get_latent_hidden(hidden_outputs)
        latent_hidden = latent_hidden.repeat(1, 1, 1)
        # latent_hidden: (layers ,batch, latent_space_size)

        return latent_repr, latent_hidden

    def _get_latent_hidden(self, hidden_output):
        latent_space_last_hidden_outputs = []
        for layer_hidden_output in hidden_output:
            latent_space_last_hidden_outputs.append(layer_hidden_output[-1])

        # There are 6 layers (3 layers for each direction), which we
        # combine for the latent hidden for the decoder
        combined = torch.cat(latent_space_last_hidden_outputs, dim=2)
        return combined

    def classify_forward(self, inputs, latent_repr):
        # Creating fake representations
        fake_inputs = create_fake_samples(inputs)
        fake_repr, _ = self.encoder_forward(fake_inputs)

        # The classifier checks if the representation is fake, so if it's fake
        # the prediction should be 1, and if it's real it should be 0
        real_repr_with_labels = [(rep, torch.tensor([0])) for rep in
                                 latent_repr]

        fake_repr_with_labels = [(rep, torch.tensor([1])) for rep in
                                 fake_repr]

        classifier_inputs = real_repr_with_labels + fake_repr_with_labels

        random.shuffle(classifier_inputs)
        combined_samples = torch.stack(
            [sample for sample, _ in classifier_inputs])
        combined_labels = torch.cat(
            [label for _, label in classifier_inputs], 0)

        classified_labels = self.classifier(combined_samples)

        classified_outputs = (classified_labels, combined_labels)
        return classified_outputs


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
        self._input_size = self._config.hidden_size[-1] * 2

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


def create_fake_sample(sample, time_steps_to_shuffle):
    fake_sample = torch.clone(sample)

    # The indices to shuffle
    indices_to_shuffle = random.sample(range(fake_sample.shape[1]),
                                       time_steps_to_shuffle)

    while len(indices_to_shuffle) > 1:
        # Choosing the indices to shuffle (The last one with a random one)
        last_index = indices_to_shuffle.pop()  # Decrease the length by 1
        random_index_to_swap = indices_to_shuffle[
            random.randint(0, len(indices_to_shuffle) - 1)]

        # Swapping the items
        swap_temp = torch.index_select(
            fake_sample, 1, torch.tensor(last_index))

        fake_sample[0, last_index] = torch.index_select(
            fake_sample, 1, torch.tensor(random_index_to_swap))

        fake_sample[0, random_index_to_swap] = swap_temp

    return fake_sample


def create_fake_samples(samples, fake_alpha=FAKE_SAMPLE_ALPHA):
    fake_samples = []

    # samples of shape [batch, time steps, single step]
    for single_sample in torch.split(samples, 1):
        # The number of samples from the series to shuffle
        time_steps_to_shuffle = math.floor(len(single_sample) * fake_alpha)
        fake_samples.append(
            create_fake_sample(single_sample, time_steps_to_shuffle))

    return torch.cat(fake_samples)


def main():
    print("Testing the DTCR functionality...")
    print("DTCR functionality finished testing.")


if __name__ == "__main__":
    main()
