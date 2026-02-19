import matplotlib.pyplot as plt
import random
import numpy as np

def plot_examples(data,cols=6,rows=5):
    figure = plt.figure(figsize=(8 * cols, 6 * rows))
    for i in range(1, cols * rows + 1):
        sample_idx = random.randint(0,len(data))
        img, label = data[sample_idx]
        img = img.permute(1,2,0)
        figure.add_subplot(rows, cols, i)
        plt.axis("off")
        plt.imshow(img)
    plt.tight_layout()
    plt.show()

def plot_clusters(tsne_embedding_2d, cluster_labels, block=True):
    plt.figure()
    plt.scatter(tsne_embedding_2d[:, 0], tsne_embedding_2d[:, 1], c=cluster_labels)
    plt.title("ImageSSL Embeddings Clustering")
    plt.show(block=block)
    plt.pause(0.1)



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