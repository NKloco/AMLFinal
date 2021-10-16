"""
This module will be the innovative part implementation of making a dtcr model
which encoder is based on an attention model instead of a bidirectional dilated
RNN.
Basically, we can use the same class, with only changing the encoder initiation
and the forward usage.
"""

import torch

from torch import nn
from Utilities.DTCR import DTCRModel


class ADTCRModel(DTCRModel):
    def get_encoder(self):
        return AttentionEncoder(self._config.input_size,
                                self._config.num_steps,
                                self._config.attention_layers,
                                sum(self._config.hidden_size)*2,
                                ff_dim=self._config.attention_ff_dim,
                                n_heads=self._config.num_heads)

    def encoder_forward(self, inputs):
        output = self.encoder(inputs)
        # output of shape [batch size, num steps, hidden size]
        # we need to reduce it to [batch size, hidden size], for that I decided
        # to avarage the representations over the steps
        output = torch.mean(output, 1)

        # Sending it twice for backwards compatibility with the BDRNN
        return output, output


class AttentionEncoder(nn.Module):
    def __init__(self, input_dim, n_steps, n_layers, dim,
                 ff_dim=1024, n_heads=1):
        super(AttentionEncoder, self).__init__()
        self.n_steps = n_steps

        self.input_embedding = nn.Linear(input_dim, dim)
        self.positional_embedding = nn.Embedding(n_steps, dim)

        self.layers = nn.ModuleList([AttentionEncoderLayer(
            dim, ff_dim, n_heads) for _ in range(n_layers)])

        self.scale = torch.sqrt(torch.FloatTensor([dim]))

    def forward(self, x):
        # x of shape [batch, n_steps, input_dim]
        # we need to combine n_steps and input_dim
        batch_size = x.shape[0]

        pos = torch.arange(0, self.n_steps).unsqueeze(0).repeat(
            batch_size, 1)

        output = self.input_embedding(x) * self.scale +\
            self.positional_embedding(pos)

        for layer in self.layers:
            output = layer(output)

        return output


class AttentionEncoderLayer(nn.Module):
    def __init__(self, dimensions, ff_dim, n_heads):
        super(AttentionEncoderLayer, self).__init__()
        self.mha = nn.MultiheadAttention(
            dimensions,
            n_heads,
            batch_first=True)

        self.first_norm = nn.LayerNorm(dimensions)

        # Fully connected feed forward
        self.ff = nn.Sequential(
            nn.Linear(dimensions, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, dimensions)
        )

        self.second_norm = nn.LayerNorm(dimensions)

    def forward(self, x):
        # Passing the multihead attention layer
        mh_output, _ = self.mha(x, x, x)

        # Passing the first normalization layer of the attention output
        # and the input
        output = self.first_norm(x + mh_output)

        # Passing through the feed forward
        ff_output = self.ff(output)

        # Second normalization
        output = self.second_norm(output + ff_output)

        return output
