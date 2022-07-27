"""Define, train, and evaluate the model from the 2015 paper "Show and Tell."
"""

import numpy as np
import torchvision.models as models
import torch.nn as nn


class CaptionModel(nn.Module):
    """Define an end-to-end image captioning model."""

    def __init__(self, num_cnn_classes, rnn_memory_size):
        """Inits the CNN and RNN components of the model."""

        super(CaptionModel, self).__init__()
        self.cnn = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.rnn = nn.LSTM(
            input_size=num_cnn_classes, hidden_size=rnn_memory_size)

    def evaluate(self, test_data, test_labels):
        """Compute the BLEU score of the model."""

        return np.NaN

    def forward(self):
        """Define the forward pass of the model."""

        pass

    def train(self):
        """Train the model."""

        pass
