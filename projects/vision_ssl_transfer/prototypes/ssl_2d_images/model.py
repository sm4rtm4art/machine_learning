import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class Model(nn.Module):
    """
    Torch nn-model wrapper class designed for 2D image inputs.
    Allows quick build of typical pipelines, e.g.,

    In -> FFT or Conv2d + MaxPool -> Linear -> Out

    Use FFT if you don't care about position of feature.
    Use FFT OR convolution for typical images.
    For images with a lot of periodic patterns over
    the full range, both FFT AND convolution
    might be the right choice.

    :param image_shape: [channels, width, height]
    :param convolution_channels:
    :param hidden_dims:
    :param embedding_dim:
    :param use_fft:
    """

    def __init__(
            self,
            image_shape: list[int],
            convolution_channels: list[int],
            hidden_dims: list[int],
            embedding_dim: int,
            use_fft=True,
            checkpoint_file=None
    ):
        super().__init__()
        self.input_channels = image_shape[0]
        self.convolution_channels = convolution_channels
        self.linear_input_dim = None  # to be determined below
        self.hidden_dims = hidden_dims
        self.embedding_dim = embedding_dim
        self.use_fft = use_fft

        self.checkpoint_file = checkpoint_file

        self.convolution_layers = nn.ModuleList()
        self.linear_layers = nn.ModuleList()

        # build network
        self.build_convolution()
        #  pass probe_image to convolution layers
        #  to get input dim for first linear layer
        #  note: current fft method doesn't change shape
        probe_tensor = torch.zeros(1, *image_shape)
        if use_fft:
            probe_tensor = self.forward_fft(probe_tensor)
        probe_tensor = self.forward_convolution(probe_tensor)
        self.linear_input_dim = math.prod(probe_tensor.shape)
        self.build_linear()

        self.property_dict = {
            "input_channels": self.input_channels,
            "convolution_channels": self.convolution_channels,
            "linear_input_dim": self.linear_input_dim,
            "hidden_dims": self.hidden_dims,
            "embedding_dim": self.embedding_dim,
            "use_fft": self.use_fft,
            "checkpoint_file": self.checkpoint_file,
        }

    # ##############
    #  network model building Methods
    #  consider making them static or move to ssl-class
    # ##############

    def build_convolution(self):
        input_channels = self.input_channels
        for convolution_channels in self.convolution_channels:
            self.add_convolution_layer(input_channels, convolution_channels)
            input_channels = convolution_channels

    def build_linear(self):
        input_dim = self.linear_input_dim
        for hidden_dim in self.hidden_dims:
            self.add_linear_layer(input_dim, hidden_dim)
            input_dim = hidden_dim

        self.linear_layers.append(nn.Linear(input_dim, self.embedding_dim))

    def add_convolution_layer(self, in_channels, out_channels, activation=nn.ReLU()):

        self.convolution_layers.append(nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(2, 2),
            padding=(1, 1)
        ))

        self.convolution_layers.append(activation)

        self.convolution_layers.append(nn.MaxPool2d(
            kernel_size=(2, 2),
            stride=(2, 2)
        ))

    def add_linear_layer(self, in_dim, out_dim, activation=nn.ReLU()):
        self.linear_layers.append(nn.Linear(in_dim, out_dim))
        if activation:
            self.linear_layers.append(activation)

    # ##############
    # forwards
    # ##############
    def forward(self, x):
        if self.use_fft:
            x = self.forward_fft(x)
        x = self.forward_convolution(x)
        x = self.forward_linear(x)
        # x = F.normalize(x, dim=1)
        # x = torch.log(1 + x)
        return F.normalize(x, dim=1)

    def forward_convolution(self, x):
        for layer in self.convolution_layers:
            x = layer(x)

        return torch.flatten(x, 1)

    def forward_linear(self, x):
        for layer in self.linear_layers:
            x = layer(x)
        return x

    # FFT is part preprocessing in model for now

    def forward_fft(self, x):
        x = torch.abs(torch.fft.fft2(x, dim=(-2, -1)))
        return x
