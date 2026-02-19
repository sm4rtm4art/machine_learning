import matplotlib.pyplot as plt
import numpy as np


def plot_examples(samples, block=True):
    # 1. Create a grid of subplots
    nrows = 6
    ncols = 8
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, 10))

    # 2. Flatten the axes array for easy iteration
    axes = axes.flatten()

    # 3. Loop through the first 16 samples
    for i in range(nrows * ncols):
        # Display the sample as an image
        axes[i].imshow(samples[i], cmap="viridis")  # 'gray' is also popular for 2D data

        # Optional: Clean up the presentation
        axes[i].axis("off")  # Hides the x/y ticks for a cleaner look

    # 4. Adjust layout to prevent title overlapping
    plt.tight_layout()
    plt.show()


#
def plot_clusters(tsne_embedding_2d, cluster_labels, block=True):
    plt.figure()
    plt.scatter(tsne_embedding_2d[:, 0], tsne_embedding_2d[:, 1], c=cluster_labels)
    plt.title("ImageSSL Embeddings Clustering")
    plt.show(block=block)
    plt.pause(0.1)


# def plot_cluster_examples(samples,cluster_labels,n_examples_per_cluster=5,block=True):
#     unique_clusters = np.unique(cluster_labels)
#
#     for cluster_id in unique_clusters:
#         indices = np.where(cluster_labels == cluster_id)[0]
#         title = f"Cluster {cluster_id} Examples"
#         plot_examples(samples[indices], n_examples_per_cluster, title, block=block)
#
#
def plot_cluster_examples(samples, cluster_labels, n_examples_per_cluster=5, block=True):
    unique_clusters = np.unique(cluster_labels)
    n_clusters = len(unique_clusters)

    # 1. Setup the grid: Rows = Clusters, Cols = Examples
    fig, axes = plt.subplots(
        n_clusters, n_examples_per_cluster, figsize=(n_examples_per_cluster * 2, n_clusters * 2)
    )

    # Handle the case where there is only 1 cluster (axes wouldn't be 2D)
    if n_clusters == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, cluster_id in enumerate(unique_clusters):
        # Get indices for this specific cluster
        indices = np.where(cluster_labels == cluster_id)[0]
        # Pick the first N examples (or use np.random.choice for random ones)
        selected_indices = indices[:n_examples_per_cluster]

        for j in range(n_examples_per_cluster):
            ax = axes[i, j]
            if j < len(selected_indices):
                # Plotting logic (assuming 1D signal data)
                ax.imshow(samples[selected_indices[j]])

                # Only put titles on the first column to save space
                if j == 0:
                    ax.set_ylabel(f"Cluster {cluster_id}", fontsize=10, rotation=0, labelpad=40)

            # Clean up the look
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    plt.show(block=block)
