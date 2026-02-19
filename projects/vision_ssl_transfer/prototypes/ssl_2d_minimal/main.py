from plot import *
from samples import generate
from ssl2d import *

"""
minimal ImageSSL testing with 1D functions:
data generation,
training,
clustering and 2D-embedding
visualization to evaluate
"""


#
# generate data
sample_number = 2048
sample_length = 128
n_features = 8
noise_strength = 0.05

training_samples = generate(n_features, sample_length, sample_number, noise_strength)
eval_samples = generate(n_features, sample_length, sample_number, noise_strength)
plot_examples(training_samples, "some training samples")


#
# initialize model
embedding_dim = 2 * n_features  # should be at least of the number of features
model = Model2d(sample_length, sample_length, embedding_dim)

# train
train(model, training_samples, temperature=0.2, episodes=50, batch_size=256)
#
# 2D embedding with TSNE, basically maps on embeddings with the greatest variances
tsne_embedding_2d = tsne_embed_2d(model, eval_samples)

# get cluster labels as numbers.
# cluster number should be number of distinct features
# if the model works
cluster_labels = get_cluster_labels(tsne_embedding_2d, n_features)


plot_clusters(tsne_embedding_2d, cluster_labels)
plot_cluster_examples(eval_samples, cluster_labels, 8)
#
#
