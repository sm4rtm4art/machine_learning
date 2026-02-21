from typing import List

from model import Model
from data_handler import ImageDataset
from pathlib import Path



import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2


from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

import math

import mlflow
import mlflow.pytorch

#
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("prototype_ssl_2d_images")


class ImageSSL:
    """
    Self-supervised learning class for a given dataset of 2D images.
    Stores datasets and possibly different nn-models.

    """

    def __init__(
            self,
            img_train_dir,
            img_test_dir,
            annotations_file="annotations.txt",
            checkpoint_file=None,
            write_out_checkpoint=False,
            resume_from_checkpoint=False,
    ):
        # model (or create list of models in future)
        self.model = None


        # optimizer
        self.optimizer = None  # to be specified in train method
        self.optimizer_type = torch.optim.Adam

        # SimCLR temperature
        self.temperature = None

        # metrics to update
        self.loss = None
        self.std_embeddings = None
        # transformation
        self.augment = None

        # training and testing datasets
        self.training_set = None
        self.testing_set = None
        self.image_shape = None

        self.import_training_images(img_train_dir, annotations_file)
        self.import_testing_images(img_test_dir, annotations_file)

        #  checkpoint for saving and/or loading
        self.checkpoint = None
        self.checkpoint_file = checkpoint_file
        self.write_out_checkpoint = write_out_checkpoint
        self.resume_from_checkpoint = resume_from_checkpoint
        self.mlflow_log_model = True

        self.check_if_checkpoint_file_exists()



    def check_if_checkpoint_file_exists(self):
        if self.checkpoint_file:
            if not Path(self.checkpoint_file).exists():
                print("Checkpoint file " + str(self.checkpoint_file) + " not found.")
                self.resume_from_checkpoint = False
                if self.write_out_checkpoint:
                    print("Use untrained model and create new checkpoint file: " + str(self.checkpoint_file))



    # DATA IMPORTS

    def import_training_images(self, img_dir, annotations_file):
        dataset = ImageDataset(img_dir,img_dir / annotations_file)
        self.training_set = dataset
        self.image_shape = dataset.image_shape
        self.set_augmentation()

    def import_testing_images(self, img_dir, annotations_file):
        dataset = ImageDataset(img_dir,img_dir / annotations_file)
        self.testing_set = dataset

    def set_model(
            self,
            convolution_channels: List[int],
            hidden_dims: List[int],
            embedding_dim: int,
            use_fft: bool,

    ):
        self.model = Model(
            image_shape=self.training_set.image_shape,
            convolution_channels=convolution_channels,
            hidden_dims=hidden_dims,
            embedding_dim=embedding_dim,
            use_fft=use_fft,
            checkpoint_file=self.checkpoint_file
        )

    # CHECKPOINT

    def save_checkpoint(self):
        self.checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "model_properties": self.model.property_dict,
            "temperature": self.temperature,
        }
        if self.write_out_checkpoint:
            torch.save(self.checkpoint, self.checkpoint_file)

    def load_checkpoint(self):
        print("Load checkpoint from " + str(self.checkpoint_file))

        self.checkpoint = torch.load(self.checkpoint_file, map_location="cpu")
        if self.model.property_dict != self.checkpoint["model_properties"]:
            raise RuntimeError("Network architecture mismatch of current and checkpoint model!")

        self.model.load_state_dict(self.checkpoint["model_state_dict"])

        # Recreate optimizer FIRST
        self.optimizer = self.optimizer_type(self.model.parameters())
        self.optimizer.load_state_dict(self.checkpoint["optimizer_state_dict"])

        self.temperature = self.checkpoint["temperature"]

        print("Checkpoint successfully loaded")
    #
    # def load_checkpoint(self):
    #
    #         print("Load checkpoint from " + str(self.checkpoint_file))
    #         self.checkpoint = torch.load(self.checkpoint_file,weights_only=False)
    #         if self.model.property_dict == self.checkpoint["model_properties"]:
    #             self.model.load_state_dict(self.checkpoint["weights_and_biases"])
    #             self.optimizer = self.checkpoint["optimizer"]
    #             self.temperature = self.checkpoint["temperature"]
    #             print("Checkpoint successfully loaded")
    #         else:
    #             raise RuntimeError("Network architecture mismatch of current and checkpoint model! ")

    # TRAINING

    def train(
            self,
            batch_size=64,
            learning_rate=1e-3,
            temperature=0.5,
            epochs=30):


        if self.resume_from_checkpoint:
            self.load_checkpoint()
        else:
            self.temperature = temperature
            self.optimizer = self.optimizer_type(self.model.parameters(), lr=learning_rate)

        image_loader = DataLoader(
            self.training_set,
            batch_size=batch_size,
            shuffle=True
        )

        loss = None
        with mlflow.start_run():
            mlflow_params = {
                "lr": learning_rate,
                "epochs": epochs,
                "batch_size": batch_size,
                "temperature": temperature
            }
            mlflow_params.update(self.model.property_dict)

            mlflow.log_params(mlflow_params)
            mlflow.config.enable_system_metrics_logging()
            mlflow.config.set_system_metrics_sampling_interval(interval=1)

            for epoch in range(epochs):

                for images, _ in image_loader:
                    images_augmented_1 = self.augment(images)
                    images_augmented_2 = self.augment(images)

                    images_embedded_1 = self.model(images_augmented_1)
                    images_embedded_2 = self.model(images_augmented_2)

                    loss = self.contrastive_loss(images_embedded_1, images_embedded_2, self.temperature)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                self.loss = loss.item()
                mlflow.log_metric("loss", self.loss, step=epoch)
                mlflow.log_metric("std_embeddings", self.std_embeddings, step=epoch)
                print(f"Epoch {epoch}, Loss {self.loss:.3f}")

                if epoch % 5 == 0:
                    self.save_checkpoint()

            #  make sure last state is saved
            self.save_checkpoint()

            if self.mlflow_log_model:
                mlflow.pytorch.log_model(self.model,name=self.checkpoint_file.replace(".","_"))

    # SimCLR stuff

    def contrastive_loss(self, images_embedded_1, images_embedded_2, temperature):
        """
        Compare similarities of augmented images already embedded by model, and compute loss.
        """
        batch_size = images_embedded_1.size(0)

        images_embedded = torch.cat([images_embedded_1, images_embedded_2], dim=0)

        self.std_embeddings = torch.mean(torch.std(images_embedded,dim=0))
        # calculate cosine similarity. Assuming image embeddings are already normalized by model ()
        # already apply temperature here to avoid duplicate multiplication in loss calculation
        similarity = torch.matmul(images_embedded, images_embedded.T) / temperature

        # remove  diagonal, i.e., set to very small nonzero value
        identity_matrix = torch.eye(2 * batch_size, dtype=torch.bool,device=similarity.device)
        similarity.masked_fill_(identity_matrix, 9e-15)

        positive_similarities = torch.cat(
            [torch.diag(similarity, batch_size), torch.diag(similarity, -batch_size)]
        )
        loss = -positive_similarities + torch.logsumexp(similarity, dim=1)
        return loss.mean()

    def set_augmentation(self):
        normalization_mean = []
        normalization_std = []

        for channel in range(self.image_shape[0]):
            normalization_mean.append(0.5)
            normalization_std.append(0.5)
        #
        self.augment = v2.Compose([
            v2.RandomHorizontalFlip(),
            v2.RandomAffine(
                degrees=(0, 15),
                translate=(0.05, 0.1),
                scale=(0.95, 1.05)),
            v2.GaussianNoise(0,0.01),
        ])

        # self.augment = v2.AutoAugment()

    # cluster estimations

    def tsne_embed_2d(self):
        print("start TSNE")
        loader = DataLoader(
            self.testing_set,
            batch_size=256,
            shuffle=True
        )

        test_images, _ = next(iter(loader))

        with torch.no_grad():
            samples_embedded = self.model(test_images)

        tsne = TSNE(n_components=2, perplexity=30)
        tsne_embedding_2d = tsne.fit_transform(samples_embedded)

        return tsne_embedding_2d, test_images

    def get_cluster_labels(self, tsne_embedding_2d, n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        cluster_labels = kmeans.fit_predict(tsne_embedding_2d)

        return cluster_labels

