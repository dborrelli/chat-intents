import numpy as np
import pandas as pd
import random
import hdbscan
import umap
from tqdm.notebook import tqdm, trange


def generate_clusters(message_embeddings, n_neighbors, n_components,
                      min_cluster_size, min_samples=None, random_state=None):

    umap_embeddings = (umap.UMAP(n_neighbors=n_neighbors,
                                 n_components=n_components,
                                 metric='cosine',
                                 random_state=random_state)
                           .fit_transform(message_embeddings))

    clusters = (hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                min_samples=min_samples,
                                metric='euclidean',
                                gen_min_span_tree=True,
                                cluster_selection_method='eom')
                       .fit(umap_embeddings))

    return clusters


def score_clusters(clusters, prob_threshold=0.05):
    """
    Returns the label count and cost of a given cluster supplied from running
    hdbscan
    """

    cluster_labels = clusters.labels_
    label_count = len(np.unique(cluster_labels))
    total_num = len(clusters.labels_)
    cost = (np.count_nonzero(clusters.probabilities_ < prob_threshold)
            / total_num)

    return label_count, cost


def random_search(embeddings, space, num_evals):

    results = []

    for i in trange(num_evals):
        n_neighbors = random.choice(space['n_neighbors'])
        n_components = random.choice(space['n_components'])
        min_cluster_size = random.choice(space['min_cluster_size'])
        min_samples = random.choice(space['min_samples'])

        clusters = generate_clusters(embeddings, 
                                     n_neighbors=n_neighbors, 
                                     n_components=n_components, 
                                     min_cluster_size=min_cluster_size, 
                                     min_samples=min_samples,
                                     random_state=42)

        label_count, cost = score_clusters(clusters, prob_threshold=0.05)

        results.append([i, n_neighbors, n_components, min_cluster_size, 
                        min_samples, label_count, cost])

    result_df = pd.DataFrame(results, 
                             columns=['run_id', 'n_neighbors', 'n_components', 
                                      'min_cluster_size', 'min_samples', 
                                      'label_count', 'cost'])

    return result_df.sort_values(by='cost')
