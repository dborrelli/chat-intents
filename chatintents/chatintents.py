import random
import collections
from functools import partial

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


class ChatIntents:
    def __init__(self, message_embeddings, name):
        self.message_embeddings = message_embeddings
        self.name = name
        self.best_params = None
        self.best_clusters = None
        self.trials = None

    def generate_clusters(self, n_neighbors, n_components,
                          min_cluster_size, min_samples=None,
                          random_state=None):

        umap_embeddings = (umap.UMAP(n_neighbors=n_neighbors,
                                     n_components=n_components,
                                     metric='cosine',
                                     random_state=random_state)
                               .fit_transform(self.message_embeddings))

        clusters = (hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                    min_samples=min_samples,
                                    metric='euclidean',
                                    gen_min_span_tree=True,
                                    cluster_selection_method='eom')
                           .fit(umap_embeddings))

        return clusters

    @staticmethod
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

    def random_search(self, space, num_evals):

        results = []

        for i in trange(num_evals):
            n_neighbors = random.choice(space['n_neighbors'])
            n_components = random.choice(space['n_components'])
            min_cluster_size = random.choice(space['min_cluster_size'])
            min_samples = random.choice(space['min_samples'])

            clusters = self.generate_clusters(n_neighbors=n_neighbors,
                                              n_components=n_components,
                                              min_cluster_size=min_cluster_size,
                                              min_samples=min_samples,
                                              random_state=42)

            label_count, cost = self.score_clusters(clusters, prob_threshold=0.05)

            results.append([i, n_neighbors, n_components, min_cluster_size,
                            min_samples, label_count, cost])

        result_df = pd.DataFrame(results,
                                 columns=['run_id', 'n_neighbors',
                                          'n_components', 'min_cluster_size',
                                          'min_samples', 'label_count', 'cost']
                                 )

        return result_df.sort_values(by='cost')

    def _objective(self, params, label_lower, label_upper):

        clusters = self.generate_clusters(n_neighbors=params['n_neighbors'],
                                          n_components=params['n_components'],
                                          min_cluster_size=params['min_cluster_size'],
                                          min_samples=params['min_samples'],
                                          random_state=params['random_state'])

        label_count, cost = self.score_clusters(clusters, prob_threshold=0.05)

        # 15% penalty on the cost function if outside the desired range
        # for the number of clusters
        if (label_count < label_lower) | (label_count > label_upper):
            penalty = 0.15
        else:
            penalty = 0

        loss = cost + penalty

        return {'loss': loss, 'label_count': label_count, 'status': STATUS_OK}

    def bayesian_search(self,
                        space,
                        label_lower,
                        label_upper,
                        max_evals=100):

        trials = Trials()
        fmin_objective = partial(self._objective,
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

        best_clusters = self.generate_clusters(n_neighbors=best_params['n_neighbors'],
                                               n_components=best_params['n_components'],
                                               min_cluster_size=best_params['min_cluster_size'],
                                               min_samples=best_params['min_samples'],
                                               random_state=best_params['random_state'])

        self.best_params = best_params
        self.best_clusters = best_clusters
        self.trials = trials
        # return best_params, best_clusters, trials

    def plot_best_clusters(self, n_neighbors=15, min_dist=0.1):
        umap_reduce = (umap.UMAP(n_neighbors=n_neighbors,
                                 n_components=2,
                                 min_dist=min_dist,
                                 # metric='cosine',
                                 random_state=42)
                           .fit_transform(self.message_embeddings)
                      )

        point_size = 100.0 / np.sqrt(self.message_embeddings.shape[0])

        result = pd.DataFrame(umap_reduce, columns=['x', 'y'])
        result['labels'] = self.best_clusters.labels_

        fig, ax = plt.subplots(figsize=(14, 8))
        noise = result[result.labels == -1]
        clustered = result[result.labels != -1]
        plt.scatter(noise.x, noise.y, color='lightgrey', s=point_size)
        plt.scatter(clustered.x, clustered.y, c=clustered.labels,
                    s=point_size, cmap='jet')
        plt.colorbar()
        plt.show()

    @staticmethod
    def _get_group(df, category_col, category):

        single_category = (df[df[category_col] == category]
                           .reset_index(drop=True)
                           )

        return single_category

    @staticmethod
    def _most_common(lst, n_words):

        counter = collections.Counter(lst)

        return counter.most_common(n_words)

    def _extract_labels(self, category_docs, print_word_counts=False):
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
            verb = self._most_common(verbs, 1)[0][0]

        if len(dobjs) > 0:
            dobj = self._most_common(dobjs, 1)[0][0]

        if len(nouns) > 0:
            noun1 = self._most_common(nouns, 1)[0][0]

        if len(set(nouns)) > 1:
            noun2 = self._most_common(nouns, 2)[1][0]

        words = [verb, dobj]

        for word in [noun1, noun2]:
            if word not in words:
                words.append(word)

        if '' in words:
            words.remove('')

        label = '_'.join(words)

        return label

    def apply_and_summarize_labels(self, df_data):

        # create a dataframe with cluster numbers applied to each doc
        category_col = 'label_' + self.name
        df_clustered = df_data.copy()
        df_clustered[category_col] = self.best_clusters.labels_

        numerical_labels = df_clustered[category_col].unique()

        # create dictionary mapping the numerical category to the generated label
        label_dict = {}
        for label in numerical_labels:
            current_category = list(self._get_group(df_clustered, category_col,
                                                    label)['text'])
            label_dict[label] = self._extract_labels(current_category)

        # create summary dataframe of numerical labels and counts
        df_summary = (df_clustered.groupby(category_col)['text'].count()
                      .reset_index()
                      .rename(columns={'text': 'count'})
                      .sort_values('count', ascending=False))

        # apply generated labels
        df_summary['label'] = df_summary.apply(lambda x:
                                               label_dict[x[category_col]],
                                               axis=1)

        labeled_docs = pd.merge(df_clustered,
                                df_summary[[category_col, 'label']],
                                on=category_col,
                                how='left')

        return df_summary, labeled_docs


def _combine_results(data_df, model_lst):
    """
    Arguments:
        data_df: dataframe of original documents with associated ground truth
                 labels
        model_lst: list of model ChatIntent instances to include in evaluation

    Returns:
        df_combined: dataframe of all documents with labels from
                     best clusters for each model

    """

    df_combined = data_df.copy()

    for model in model_lst:
        label_name = 'label_' + model.name
        df_combined[label_name] = model.best_clusters.labels_

    return df_combined


def evaluate_models(data_df, model_lst):
    """
    Arguments:
        data_df: dataframe of original documents with associated ground truth
                 labels
        model_lst: list of model ChatIntent instances to include in evaluation

    Returns:
        labeled_docs_all_models: dataframe of all documents with labels from
                                 best clusters for each model

    """

    df_combined = _combine_results(data_df, model_lst)

    summary = []

    for model in model_lst:
        ground_labels = df_combined['category'].values
        clustered_labels = df_combined['label_' + model.name].values

        ari = np.round(adjusted_rand_score(ground_labels,
                                           clustered_labels), 3)
        nmi = np.round(normalized_mutual_info_score(ground_labels,
                                                    clustered_labels), 3)
        summary.append([model.name, ari, nmi])

    df_evaluation = pd.DataFrame(summary, columns=['Model', 'ARI', 'NMI'])

    return df_evaluation.sort_values(by='NMI', ascending=False), df_combined


def top_cluster_category(df_clusters, df_ground, key, df_summary):

    df_label = pd.merge(df_clusters, df_ground, on=key, how='left')

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
