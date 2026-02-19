import numpy as np

from image_ssl import ImageSSL
from plot import plot_examples, plot_clusters, plot_cluster_examples
from generate_images import generate_and_save_images

image_number = 1024
image_width = 128
n_features = 8
img_train_dir, img_test_dir, annotations_file_name \
    = generate_and_save_images(
    sample_number=image_number,
    sample_length=image_width,
    n_features=8
)

ssl = ImageSSL()
ssl.import_training_images(
    img_train_dir,
    img_train_dir / annotations_file_name
)
ssl.import_testing_images(
    img_test_dir,
    img_test_dir / annotations_file_name
)

ssl.set_model(
    convolution_channels=[8,16,32],
    hidden_dims=[128],
    embedding_dim=32,
    use_fft=False
)

ssl.train(epochs=500, batch_size=32,learning_rate=1e-3)
tsne_embedding_2d, test_images = ssl.tsne_embed_2d()
cluster_labels = ssl.get_cluster_labels(tsne_embedding_2d, n_features)
plot_clusters(tsne_embedding_2d, cluster_labels)

plot_cluster_examples(test_images.permute(0, 2, 3, 1), cluster_labels)
