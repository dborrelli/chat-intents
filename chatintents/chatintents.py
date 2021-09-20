import random
from functools import partial
import collections

import numpy as np
import pandas as pd
import hdbscan
import umap
from tqdm.notebook import tqdm, trange
from hyperopt import fmin, tpe, hp, STATUS_OK, space_eval, Trials

from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
import matplotlib.pyplot as plt
import spacy


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


def objective(params, embeddings, label_lower, label_upper):

    clusters = generate_clusters(embeddings,
                                 n_neighbors=params['n_neighbors'],
                                 n_components=params['n_components'],
                                 min_cluster_size=params['min_cluster_size'],
                                 min_samples=params['min_samples'],
                                 random_state=params['random_state'])

    label_count, cost = score_clusters(clusters, prob_threshold=0.05)

    # 15% penalty on the cost function if outside the desired range of groups
    if (label_count < label_lower) | (label_count > label_upper):
        penalty = 0.15
    else:
        penalty = 0

    loss = cost + penalty

    return {'loss': loss, 'label_count': label_count, 'status': STATUS_OK}


def bayesian_search(embeddings,
                    space,
                    label_lower,
                    label_upper,
                    max_evals=100):

    trials = Trials()
    fmin_objective = partial(objective,
                             embeddings=embeddings,
                             label_lower=label_lower,
                             label_upper=label_upper)

    best = fmin(fmin_objective,
                space=space,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials)

    best_params = space_eval(space, best)
    print('best:')
    print(best_params)
    print(f"label count: {trials.best_trial['result']['label_count']}")

    best_clusters = generate_clusters(embeddings,
                                      n_neighbors=best_params['n_neighbors'],
                                      n_components=best_params['n_components'],
                                      min_cluster_size=best_params['min_cluster_size'],
                                      min_samples=best_params['min_samples'],
                                      random_state=best_params['random_state'])

    return best_params, best_clusters, trials


def combine_results(data_df, cluster_dict):

    df = data_df.copy()

    for key, value in cluster_dict.items():
        df[key] = value.labels_

    return df


def comparison_table(model_dict, results_df):
    summary = []

    for key, value in model_dict.items():
        ground_labels = results_df['category'].values
        clustered_labels = results_df[value].values

        ari = np.round(adjusted_rand_score(ground_labels,
                                           clustered_labels), 3)
        nmi = np.round(normalized_mutual_info_score(ground_labels,
                                                    clustered_labels), 3)
        summary.append([key, ari, nmi])

    comparison_df = pd.DataFrame(summary, columns=['Model', 'ARI', 'NMI'])

    return comparison_df.sort_values(by='NMI', ascending=False)


def plot_clusters(embeddings, clusters, n_neighbors=15, min_dist=0.1):
    umap_data = umap.UMAP(n_neighbors=n_neighbors,
                          n_components=2,
                          min_dist=min_dist,
                          # metric='cosine',
                          random_state=42).fit_transform(embeddings)

    point_size = 100.0 / np.sqrt(embeddings.shape[0])

    result = pd.DataFrame(umap_data, columns=['x', 'y'])
    result['labels'] = clusters.labels_

    fig, ax = plt.subplots(figsize=(14, 8))
    outliers = result[result.labels == -1]
    clustered = result[result.labels != -1]
    plt.scatter(outliers.x, outliers.y, color='lightgrey', s=point_size)
    plt.scatter(clustered.x, clustered.y, c=clustered.labels,
                s=point_size, cmap='jet')
    plt.colorbar()
    plt.show()


def get_group(df, category_col, category):

    single_category = df[df[category_col] == category].reset_index(drop=True)

    return single_category


def most_common(lst, n_words):
    counter = collections.Counter(lst)

    return counter.most_common(n_words)


def extract_labels(category_docs, print_word_counts=False):
    """
    Argument:
    category_docs: list of documents, all from the same category or clustering
    """
    verbs = []
    dobjs = []
    nouns = []
    adjs = []

    verb = ''
    dobj = ''
    noun1 = ''
    noun2 = ''

    nlp = spacy.load("en_core_web_sm")

    for i in range(len(category_docs)):
        doc = nlp(category_docs[i])
        for token in doc:
            if token.is_stop is False:
                if token.dep_ == 'ROOT':
                    verbs.append(token.text.lower())

                elif token.dep_ == 'dobj':
                    dobjs.append(token.lemma_.lower())

                elif token.pos_ == 'NOUN':
                    nouns.append(token.lemma_.lower())

                elif token.pos_ == 'ADJ':
                    adjs.append(token.lemma_.lower())

    if print_word_counts:
        for word_lst in [verbs, dobjs, nouns, adjs]:
            counter = collections.Counter(word_lst)
            print(counter)

    if len(verbs) > 0:
        verb = most_common(verbs, 1)[0][0]

    if len(dobjs) > 0:
        dobj = most_common(dobjs, 1)[0][0]

    if len(nouns) > 0:
        noun1 = most_common(nouns, 1)[0][0]

    if len(set(nouns)) > 1:
        noun2 = most_common(nouns, 2)[1][0]

    words = [verb, dobj]

    for word in [noun1, noun2]:
        if word not in words:
            words.append(word)

    if '' in words:
        words.remove('')

    label = '_'.join(words)

    return label


def apply_and_summarize_labels(df, category_col):
    numerical_labels = df[category_col].unique()

    # create dictionary of the numerical category to the generated label
    label_dict = {}
    for label in numerical_labels:
        current_category = list(get_group(df, category_col, label)['text'])
        label_dict[label] = extract_labels(current_category)

    # create summary dataframe of numerical labels and counts
    summary_df = (df.groupby(category_col)['text'].count()
                    .reset_index()
                    .rename(columns={'text': 'count'})
                    .sort_values('count', ascending=False))

    # apply generated labels
    summary_df['label'] = summary_df.apply(lambda x:
                                           label_dict[x[category_col]],
                                           axis=1)

    return summary_df


def combine_ground_truth(df_clusters, df_ground, key):
    df_combined = pd.merge(df_clusters, df_ground, on=key, how='left')
    return df_combined


def get_top_category(df_label, df_summary):
    df_label_ground = (df_label.groupby('label')
                       .agg(top_ground_category=('category',
                                                 lambda x:
                                                 x.value_counts().index[0]),
                            top_cat_count=('category',
                                           lambda x: x.value_counts()[0]))
                       .reset_index())

    df_result = pd.merge(df_summary, df_label_ground, on='label', how='left')
    df_result['perc_top_cat'] = df_result.apply(lambda x:
                                                int(round(100 * x['top_cat_count']/ x['count'])),
                                                axis=1)

    return df_result