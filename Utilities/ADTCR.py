"""
This module will be the innovative part implementation of making a dtcr model
which encoder is based on an attention model instead of a bidirectional dilated
RNN.
Basically, we can use the same class, with only changing the encoder initiation
and the forward usage.
"""

from torch import nn
from Utilities.DTCR import DTCRModel


class ADTCRModel(DTCRModel):
    def get_encoder(self):
        return nn.MultiheadAttention(
            self._config.input_size,
            self._config.num_heads,
            batch_first=True)

    def encoder_forward(self, inputs):
        return super().encoder_forward(inputs)