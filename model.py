"""Define an image captioning model."""


import torchvision.models as cv_models
import torch.nn as nn_models


class Model:
    """Define an end-to-end image captioning model."""

    def __init__(self, num_cnn_classes, rnn_memory_size):
        """Inits the CNN and RNN components of the model."""

        self.cnn = cv_models.resnet50(
            weights=cv_models.ResNet50_Weights.DEFAULT)
        self.rnn = nn_models.LSTM(
            input_size=num_cnn_classes, hidden_size=rnn_memory_size)
