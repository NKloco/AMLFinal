"""
This module will provide the functionality required to recreate the results of
the deep temporal clustering representation(DTCR) algorithm.
"""
import math
import random
import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import rand_score, normalized_mutual_info_score

from Utilities.DRNN import BidirectionalDRNN

FAKE_SAMPLE_ALPHA = 0.3  # As set in the article
DEFAULT_COLORS = ["red", "green", "blue", "purple", "orange", "yellow"]


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
    training_printing_interval = 1
    checkpoint_interval = 30
    checkpoint_path = "Checkpoints"
    model_name = None
    optimizer = torch.optim.Adam

    # Encoder Settings
    hidden_size = [100, 50, 50]
    dilations = [1, 2, 4]
    encoder_cell_type = 'GRU'

    # Decoder Settings
    decoder_cell_type = 'GRU'
    decoding_criterion = nn.MSELoss

    # Classifier Settings
    classifier_nodes = 128
    classifier_criterion = nn.CrossEntropyLoss

    # Clustering Settings
    f_update_interval = 10
    coefficient_lambda = 0.1


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

        F = torch.empty(self._config.batch_size, self._config.class_num)
        self.F = torch.nn.init.orthogonal_(F)

        self._training_iteration = 0

    def train_step(self, train_dl, test_dl,
                   recons_criterion, classify_criterion, optimizer):
        # This is used for the F update, model checkpoint, and graph making
        self._training_iteration += 1
        iteration = self._training_iteration
        should_update_F = (iteration % self._config.f_update_interval == 0)

        print_interval = self._config.training_printing_interval
        clustering_lambda = self._config.coefficient_lambda
        running_loss = 0.0
        running_recons_loss = 0.0
        running_classify_loss = 0.0
        running_clustering = 0.0

        # Training over the data once
        for index, (sample_data, _) in enumerate(train_dl):
            # Zeros the gradients for the next forward
            optimizer.zero_grad()

            # Forward of the model, makes latent space representations,
            # creates fake samples and classify them, anc calculates the
            # clustering loss
            _, reconstructed_inputs, classified_outputs, clustering_loss\
                = self(sample_data, should_update_f=should_update_F)

            recons_loss = recons_criterion(reconstructed_inputs, sample_data)
            classify_loss = classify_criterion(*classified_outputs)

            # The clustering loss contributes only lambda to the whole loss
            lambda_clustering_loss = clustering_loss * clustering_lambda
            dtcr_loss = recons_loss + classify_loss + lambda_clustering_loss

            # Backwards of the network
            dtcr_loss.backward()

            # Optimization step
            optimizer.step()

            # Information logging
            running_recons_loss += recons_loss.item()
            running_classify_loss += classify_loss.item()
            running_clustering += clustering_loss.item()
            running_loss += dtcr_loss.item()
            if index % print_interval == print_interval - 1:
                print(('[{}|{}] loss: {:.4f}, classify: {:.5f},' +
                       'recons: {:.5f}, clustering: {:.5f}').format(
                        iteration, index + 1,
                        running_loss / print_interval,
                        running_classify_loss / print_interval,
                        running_recons_loss / print_interval,
                        running_clustering / print_interval))
                running_loss = 0.0
                running_classify_loss = 0.0
                running_recons_loss = 0.0
                running_clustering = 0.0

        # After training, we might want to test
        if iteration % self._config.checkpoint_interval == 0:
            self._make_checkpoint()
            with torch.no_grad():
                for test_data, test_label in test_dl:
                    self._evaluate_model(test_data, test_label)

    def forward(self, inputs, should_update_f=False):
        # inputs of shape (Batch, Time Steps, Single step size)
        latent_repr, _ = self.encoder_forward(inputs)

        # For decoding we also need a time-step dimension (which is 1)
        # reconstructed_inputs will have the same shape as the inputs
        decoder_input = latent_repr.expand(1, -1, -1).transpose(0, 1)
        decoder_hidden = latent_repr.expand(1, -1, -1)
        reconstructed_inputs = self.decoder(decoder_input,
                                            hidden=decoder_hidden)

        classified_outputs = self.classify_forward(inputs, latent_repr)

        clustering_loss = self._calculate_clustering_loss(
            latent_repr, should_update=should_update_f)

        return latent_repr, reconstructed_inputs, classified_outputs,\
            clustering_loss

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

    def _calculate_clustering_loss(self, representations,
                                   should_update=False):
        # representations of shape [batch, latent space size]
        # H will be of size [latent space, batch] as in the paper
        H = representations.transpose(0, 1)
        HTH = torch.matmul(H.transpose(0, 1), H)
        FTHTH = torch.matmul(self.F.transpose(0, 1), HTH)
        FTHTHF = torch.matmul(FTHTH, self.F)

        clustering_loss = torch.trace(HTH) - torch.trace(FTHTHF)

        if should_update:
            np_H = H.detach().numpy()
            U, sigma, VT = np.linalg.svd(np_H)
            sorted_indices = np.argsort(sigma)
            k_evecs = VT[sorted_indices[:-self._config.class_num - 1:-1], :]
            self.F = torch.from_numpy(k_evecs.T)

        return clustering_loss

    def _make_checkpoint(self):
        checkpoint_name = "{}_{}".format(self._config.model_name,
                                         self._training_iteration)
        path = os.path.join(self._config.checkpoint_path, checkpoint_name)
        torch.save(self, path)

    def _evaluate_model(self, test_data, test_labels):
        print("Evaluating Model:")
        true_labels = test_labels.numpy()

        # If the labels are not 0 based I change them to be
        if 0 not in true_labels:
            true_labels -= 1
        data_repr, _ = self.encoder_forward(test_data)
        data_repr_numpy = data_repr.numpy()
        kmeans = KMeans(n_clusters=self._config.class_num).fit(data_repr_numpy)
        predicted_labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        # The label is the number of classes, because the classes start at 0
        center_label = self._config.class_num
        DEFAULT_COLORS[center_label] = "black"
        center_labels = np.asarray([center_label]*self._config.class_num)

        # Calculating the rand index and normalized mutual information to
        # evaluate the model
        rand_index = rand_score(true_labels, predicted_labels)
        nmi = normalized_mutual_info_score(true_labels, predicted_labels)
        print("RI: {}, NMI: {}".format(rand_index, nmi))

        # plotting the representations with the classes and the centers
        all_data = np.concatenate([data_repr_numpy, centers])
        data_points = TSNE(
            n_components=self._config.class_num).fit_transform(all_data)

        scatter_x = data_points[:, 0]
        scatter_y = data_points[:, 1]
        all_labels = np.concatenate([true_labels, center_labels])
        all_predicted_labels = np.concatenate(
            [predicted_labels, center_labels])

        print("Plot with original labels:")
        fig, ax = plt.subplots()
        for c_label in range(center_label + 1):
            ix = np.where(all_labels == c_label)
            brush_size = 50 if c_label == center_label else 20
            ax.scatter(scatter_x[ix], scatter_y[ix], c=DEFAULT_COLORS[c_label],
                       label=c_label, s=brush_size)
        ax.legend()
        plt.show()

        print("Plot with predicted labels (for kmeans and tsne sanity):")
        fig, ax = plt.subplots()
        for c_label in range(center_label + 1):
            ix = np.where(all_predicted_labels == c_label)
            brush_size = 50 if c_label == center_label else 20
            ax.scatter(scatter_x[ix], scatter_y[ix], c=DEFAULT_COLORS[c_label],
                       label=c_label, s=brush_size)
        ax.legend()
        plt.show()


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
