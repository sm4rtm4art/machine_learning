from time import sleep

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import numpy.typing as npt
import torch
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import v2
import math

from data_handler import ImageDataset


class ImageSSL:
    """
    Self-supervised learning class for a given dataset of 2D images.
    Stores datasets and possibly different nn-models.

    """

    def __init__(self):
        # set model (or create list of models in future)
        self.model = None

        # training and testing datasets
        self.training_set = None
        self.testing_set = None
        self.image_loader = None

        # transformation
        self.augment = v2.Compose([
            v2.RandomRotation(360),
            v2.GaussianNoise(),
            v2.RandomVerticalFlip(),
            v2.ElasticTransform()
        ])

        # training stuff
        self.batch_size = 32

    def import_training_images(self, img_dir, annotations_file):
        dataset = ImageDataset(img_dir, annotations_file)
        self.training_set = dataset

    def import_testing_images(self, img_dir, annotations_file):
        dataset = ImageDataset(img_dir, annotations_file)
        self.testing_set = dataset

    def set_model(
            self,
            convolution_channels,
            hidden_dims,
            embedding_dim,
            use_fft
    ):
        self.model = Model(
            image_shape=self.training_set.image_shape,
            convolution_channels=convolution_channels,
            hidden_dims=hidden_dims,
            embedding_dim=embedding_dim,
            use_fft=use_fft
        )

    def train(
            self,
            batch_size=64,
            learning_rate=1e-3,
            temperature=0.5,
            epochs=30):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        image_loader = DataLoader(
            self.training_set,
            batch_size=batch_size,
            shuffle=True
        )

        for step in range(epochs):
            for images, _ in image_loader:
                images_augmented_1 = self.augment(images)
                images_augmented_2 = self.augment(images)

                images_embedded_1 = self.model(images_augmented_1)
                images_embedded_2 = self.model(images_augmented_2)

                loss = self.contrastive_loss(images_embedded_1, images_embedded_2, temperature)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # if step % 5 == 0:
            print(f"Step {step}, Loss {loss.item():.4f}")

    #

    def contrastive_loss(self, images_embedded_1, images_embedded_2, temperature):
        """
        Compare similarities of augmented images already embedded by model, and compute loss.
        """
        batch_size = images_embedded_1.size(0)
        samples_embedded = torch.cat([images_embedded_1, images_embedded_2], dim=0)

        similarity = torch.matmul(samples_embedded, samples_embedded.T) / temperature

        # remove  diagonal, i.e., set to very small nonzero value
        identity_matrix = torch.eye(2 * batch_size, dtype=torch.bool).to(samples_embedded.device)
        similarity.masked_fill_(identity_matrix, -9e15)

        positive_similarities = torch.cat(
            [torch.diag(similarity, batch_size), torch.diag(similarity, -batch_size)]
        )
        loss = -positive_similarities + torch.logsumexp(similarity, dim=1)
        return loss.mean()

    def tsne_embed_2d(self):
        print("start TSNE")
        loader = DataLoader(
            self.testing_set,
            batch_size= 256,
            shuffle=True
        )

        test_images, _ = next(iter(loader))

        with torch.no_grad():
            samples_embedded = self.model(test_images)

        tsne = TSNE(n_components=2, perplexity=30)
        tsne_embedding_2d = tsne.fit_transform(samples_embedded)

        return tsne_embedding_2d, test_images

    def get_cluster_labels(self,tsne_embedding_2d, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        cluster_labels = kmeans.fit_predict(tsne_embedding_2d)

        return cluster_labels


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

    def __init__(self,
                 image_shape: list[int],
                 convolution_channels: list[int],
                 hidden_dims: list[int],
                 embedding_dim: int,
                 use_fft=True):
        super().__init__()
        self.input_channels = image_shape[0]
        self.convolution_channels = convolution_channels
        self.hidden_dims = hidden_dims
        self.embedding_dim = embedding_dim
        self.use_fft = use_fft

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
            kernel_size=3,
            stride=2,
            padding=1
        ))

        self.convolution_layers.append(activation)


        self.convolution_layers.append(nn.MaxPool2d(
            kernel_size=2,
            stride=2
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
    # could also be outside Model class

    def forward_fft(self, x):
        x = torch.abs(torch.fft.fft2(x, dim=(-2, -1)))
        return x
