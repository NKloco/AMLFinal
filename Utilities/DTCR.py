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
    learning_rate = 5e-8
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
        latent_space_size = sum(self._config.hidden_size) * 2
        classifier = nn.Sequential(
            nn.Linear(latent_space_size, self._config.classifier_nodes),
            nn.ReLU(),
            nn.Linear(self._config.classifier_nodes, 2),
            nn.Softmax(dim=1)
        )

        return classifier

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

    def classify_forward(self, inputs, latent_repr):
        # Creating fake representations
        fake_inputs = create_fake_samples(inputs)
        _, fake_hidden_outputs = self.encoder(fake_inputs)
        fake_repr = self.get_latent_representation(fake_hidden_outputs)

        # The classifier checks if the representation is fake, so if it's fake
        # the prediction should be 1, and if it's real it should be 0
        real_repr_with_labels = [sample for sample in
                                 zip(latent_repr,
                                     [torch.tensor([1, 0])]*len(latent_repr))]

        fake_repr_with_labels = [sample for sample in
                                 zip(fake_repr,
                                     [torch.tensor([0, 1])]*len(fake_repr))]

        classifier_inputs = real_repr_with_labels + fake_repr_with_labels

        random.shuffle(classifier_inputs)
        combined_samples = torch.stack(
            [sample for sample, _ in classifier_inputs])
        combined_labels = torch.stack(
            [label for _, label in classifier_inputs])

        classified_labels = self._classifier(combined_samples)

        classified_outputs = (classified_labels, combined_labels)
        return classified_outputs

    def forward(self, inputs):
        # inputs of shape (Batch, Time Steps, Single step size)

        _, hidden_outputs = self.encoder(inputs)
        # hidden_outputs: list of length of layers * directions (6)
        # each item of shape (dilation, batch, hidden size of layer)

        latent_repr = self.get_latent_representation(hidden_outputs)
        # latent_repr: (batch, latent_space_size)

        prep_for_decoder = latent_repr.repeat(1, 1, 1).transpose(0, 1)
        # prep_for_decoder: (batch, time steps [1], input size)

        reconstructed_inputs = self.decoder(prep_for_decoder,
                                            hidden=latent_repr.repeat(1, 1, 1))
        # reconstructed_inputs have the same shape as the input

        classified_outputs = self.classify_forward(inputs, latent_repr)

        return inputs, latent_repr, reconstructed_inputs, classified_outputs

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

        self.train(mode=True)
        for index, (sample_data, sample_label) in enumerate(train_dl):
            optimizer.zero_grad()

            inputs, latent_repr, reconstructed_inputs, classified_outputs =\
                self(sample_data)

            recons_loss = recons_criterion(reconstructed_inputs, inputs)
            classify_loss = classify_criterion(
                classified_outputs[0],
                torch.max(classified_outputs[1], 1)[1])

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

        self.train(mode=False)


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
