import numpy as np

from image_ssl import ImageSSL
from plot import plot_examples, plot_clusters, plot_cluster_examples
from generate_images import generate_and_save_images
from ml_portfolio.common.paths import get_project_paths

PROJECT_NAME = "vision_ssl_transfer"

img_train_dir_base = "images_train"
img_test_dir_base = "images_test"

n_features = 8

#  script switches
regenerate_images = True
plot_training_set_examples = False
plot_augmentated_examples = False
train_model = True
plot_testing_set_clusters = True

if regenerate_images:
    image_number = 1024 *2
    image_width = 256

    img_train_dir, img_test_dir, annotations_file_name \
        = generate_and_save_images(
        img_train_dir_base=img_train_dir_base,
        img_test_dir_base=img_test_dir_base,
        sample_number=image_number,
        sample_length=image_width,
        n_features=n_features
    )
else:
    output_dir = get_project_paths(PROJECT_NAME).data_dir / "prototypes/ssl_2d_images"
    img_train_dir = output_dir / img_train_dir_base
    img_test_dir = output_dir / img_test_dir_base
    annotations_file_name = "annotations.txt"

ssl = ImageSSL()
ssl.import_training_images(
    img_train_dir,
    img_train_dir / annotations_file_name
)
ssl.import_testing_images(
    img_test_dir,
    img_test_dir / annotations_file_name
)

if plot_training_set_examples:
    plot_examples(ssl.training_set)




# test augmentation
if plot_augmentated_examples:
    from data_handler import AugmentedDataset
    augmented_set  = AugmentedDataset(ssl.training_set,ssl.augment)
    plot_examples(augmented_set)

if train_model:
    ssl.set_model(
        convolution_channels=[8,16,32,64],
        hidden_dims=[128],
        embedding_dim=n_features,
        use_fft=False
    )

    ssl.train(epochs=500, batch_size=512, learning_rate=1e-3)




    ssl.train(epochs=500, batch_size=256, learning_rate=1e-3)

if plot_testing_set_clusters:
    tsne_embedding_2d, test_images = ssl.tsne_embed_2d()
    cluster_labels = ssl.get_cluster_labels(tsne_embedding_2d, n_features)
    plot_clusters(tsne_embedding_2d, cluster_labels)

    plot_cluster_examples(test_images.permute(0, 2, 3, 1), cluster_labels)
