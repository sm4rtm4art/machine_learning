import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model1d(nn.Module):
    def __init__(self, input_dim=64, embedding_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            self.FFT(), nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, embedding_dim)
        )

    def forward(self, x):
        z = self.net(x)
        return F.normalize(z, dim=1)  # important for contrastive loss

    # fft module for model
    class FFT(nn.Module):
        def __init__(self, dim=-1):
            super().__init__()
            self.dim = dim

        def forward(self, x):
            return torch.abs(torch.fft.fft(x, dim=self.dim))


def augment(data_original):
    data_augmented = data_original.copy()

    # Add more noise
    data_augmented += 0.02 * np.random.randn(*data_original.shape)

    # Amplitude scaling
    scale = np.random.uniform(0.9, 1.1)
    data_augmented *= scale

    # Small horizontal shift
    horizonal_shift = np.random.randint(-5, 5)
    data_augmented = np.roll(data_augmented, horizonal_shift)

    # small vertical shift
    verical_shift = 0.3 * np.random.random()
    data_augmented = data_augmented + verical_shift

    return data_augmented.astype(np.float32)


"""
compare similarities of augmented already embedded data, and compute loss
"""


def contrastive_loss(samples_embedded_1, samples_embedded_2, temperature):
    batch_size = samples_embedded_1.size(0)
    samples_embedded = torch.cat([samples_embedded_1, samples_embedded_2], dim=0)

    similarity = torch.matmul(samples_embedded, samples_embedded.T) / temperature

    # remove  diagonal, i.e., set to very small nonzero value
    identity_matrix = torch.eye(2 * batch_size, dtype=torch.bool).to(samples_embedded.device)
    similarity.masked_fill_(identity_matrix, -9e15)

    positive_similarities = torch.cat(
        [torch.diag(similarity, batch_size), torch.diag(similarity, -batch_size)]
    )
    loss = -positive_similarities + torch.logsumexp(similarity, dim=1)
    return loss.mean()


def train(model, training_data, batch_size=128, learning_rate=1e-3, temperature=0.5, episodes=2000):
    sample_number = training_data.shape[0]

    if batch_size > sample_number:
        batch_size = sample_number

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    index_range = range(sample_number)
    for step in range(episodes):
        indices = np.random.choice(index_range, batch_size, replace=False)
        samples = training_data[indices]

        samples_augmented_1 = np.stack([augment(sample) for sample in samples])
        samples_augmented_2 = np.stack([augment(sample) for sample in samples])
        #
        samples_augmented_1 = torch.tensor(samples_augmented_1)
        samples_augmented_2 = torch.tensor(samples_augmented_2)
        #
        samples_embedded_1 = model(samples_augmented_1)
        samples_embedded_2 = model(samples_augmented_2)

        loss = contrastive_loss(samples_embedded_1, samples_embedded_2, temperature)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 200 == 0:
            print(f"Step {step}, Loss {loss.item():.4f}")


def tsne_embed_2d(model, eval_samples):
    eval_samples = torch.tensor(eval_samples.astype(np.float32))

    # model.eval()
    with torch.no_grad():
        samples_embedded = model(eval_samples).numpy()

    from sklearn.manifold import TSNE

    tsne = TSNE(n_components=2, perplexity=30)
    tsne_embedding_2d = tsne.fit_transform(samples_embedded)

    return tsne_embedding_2d


def get_cluster_labels(tsne_embedding_2d, n_clusters):
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(tsne_embedding_2d)

    return cluster_labels
