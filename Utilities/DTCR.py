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
    training_printing_interval = 1
    optimizer = torch.optim.Adam

    # Encoder Settings
    hidden_size = [50, 30, 30]
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
        self.classifier = nn.Sequential(
            nn.Linear(self._latent_space_size, self._config.classifier_nodes),
            nn.ReLU(),
            nn.Linear(self._config.classifier_nodes, 2),
            nn.Softmax(dim=1)
        )

    def train_step(self, train_dl, test_dl,
                   recons_criterion, classify_criterion, optimizer):
        # Going over the batches
        print_interval = self._config.training_printing_interval
        running_loss = 0.0
        running_recons_loss = 0.0
        running_classify_loss = 0.0

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
        latent_repr, _ = self.encoder_forward(inputs)

        # For decoding we also need a time-step dimension (which is 1)
        # reconstructed_inputs will have the same shape as the inputs
        decoder_input = latent_repr.expand(1, -1, -1).transpose(0, 1)
        decoder_hidden = latent_repr.expand(1, -1, -1)
        reconstructed_inputs = self.decoder(decoder_input,
                                            hidden=decoder_hidden)

        classified_outputs = self.classify_forward(inputs, latent_repr)

        return inputs, latent_repr, reconstructed_inputs, classified_outputs

    def encoder_forward(self, inputs):
        _, hidden_outputs = self.encoder(inputs)

        latent_repr = self._get_latent_repr(hidden_outputs)
        # latent_repr: (batch, latent_space_size)

        return latent_repr, hidden_outputs

    def _get_latent_repr(self, hidden_output):
        # hidden_outputs: list of length of layers * directions (6)
        # each is a list of length dilation of the layer
        # and each item is a tensor of (batch, hidden size of layer)
        latent_space_last_hidden_outputs = []
        for layer_hidden_output in hidden_output:
            # We take the last dilation for the hidden part
            latent_space_last_hidden_outputs.append(layer_hidden_output[-1])

        combined = torch.cat(latent_space_last_hidden_outputs, dim=1)
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
    # The indices to shuffle
    indices_to_shuffle = random.sample(range(len(sample)),
                                       time_steps_to_shuffle)

    while len(indices_to_shuffle) > 1:
        # Choosing the indices to shuffle (The last one with a random one)
        last_index = indices_to_shuffle.pop()  # Decrease the length by 1
        random_index_to_swap = indices_to_shuffle[
            random.randint(0, len(indices_to_shuffle) - 1)]

        # Swapping the items
        swap_temp = sample[last_index]
        sample[last_index] = sample[random_index_to_swap]
        sample[random_index_to_swap] = swap_temp

    return sample


def create_fake_samples(samples, fake_alpha=FAKE_SAMPLE_ALPHA):
    fake_samples = []
    new_samples = samples.tolist()
    # samples of shape [batch, time steps, single step]
    for single_series in new_samples:
        # The number of samples from the series to shuffle
        time_steps_to_shuffle = math.floor(len(single_series) * fake_alpha)
        fake_samples.append(
            create_fake_sample(single_series, time_steps_to_shuffle))

    return torch.tensor(fake_samples)


def main():
    print("Testing the DTCR functionality...")
    # the fake sample test is like a batch of 3 time series of 10 time steps
    # with 1 dimensional sampling for each timestep.
    fake_samples_test = torch.tensor([
        [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]],
        [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]],
        [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]],
    ])
    fake_samples = create_fake_samples(fake_samples_test)
    print("Original shape: {}, Fake shape: {}".format(
        fake_samples_test.shape, fake_samples.shape))
    for batch_instance in fake_samples:
        print(",".join(
            [str(time_step.item()) for time_step in batch_instance]))
    print("DTCR functionality finished testing.")


if __name__ == "__main__":
    main()
