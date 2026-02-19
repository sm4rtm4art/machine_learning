"""
minimal ImageSSL testing with 1D functions:
data generation,
training,
clustering and 2D-embedding
visualization to evaluate
"""

from plot import plot_cluster_examples, plot_clusters, plot_examples
from samples import generate
from ssl1d import Model1d, get_cluster_labels, train, tsne_embed_2d

# generate data
sample_number = 1024
sample_length = 258
n_features = 4  # number of distinct features to generate
noise_strength = 0.1
training_samples = generate(sample_length, sample_number, n_features, noise_strength)
eval_samples = generate(sample_length, sample_number, n_features, noise_strength)
plot_examples(training_samples, title="some training samples")


# initialize model
embedding_dim = 4  # should be at least of the number of features
temperature = 0.5  # soft max temperature for contrastive loss
model = Model1d(sample_length, embedding_dim)

# train
train(model, training_samples, temperature=temperature, episodes=30)

# 2D embedding with TSNE, basically maps on embeddings with the greatest
# variances
tsne_embedding_2d = tsne_embed_2d(model, eval_samples)

# get cluster labels as numbers.
# cluster number should be number of distinct features
# if the model works
cluster_labels = get_cluster_labels(tsne_embedding_2d, n_features)


plot_clusters(tsne_embedding_2d, cluster_labels)
plot_cluster_examples(eval_samples, cluster_labels, 5)
