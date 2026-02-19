from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from generate_images import generate
import torch
from plot import plot_clusters, plot_cluster_examples


def tsne_embed_2d(data):
        print("start TSNE")

        tsne = TSNE(n_components=2, perplexity=30)
        tsne_embedding_2d = tsne.fit_transform(data)

        return tsne_embedding_2d

def get_cluster_labels(tsne_embedding_2d, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(tsne_embedding_2d)

    return cluster_labels

samples = generate(
    n_features=8,
    sample_length=128,
    sample_number=1024,
    noise_strength=0.05
)


samples = torch.tensor(samples)
fft_samples = torch.abs(torch.fft.fft2(samples,dim=(-2, -1)))
fft_samples_flat = torch.flatten(fft_samples,-2,-1)
samples_2d = tsne_embed_2d(fft_samples_flat)
cluster_labels = get_cluster_labels(samples_2d,8)
plot_clusters(samples_2d,cluster_labels)
plot_cluster_examples(samples,cluster_labels,20)


