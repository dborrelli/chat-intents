import random
import collections
from functools import partial

import numpy as np
import pandas as pd
import hdbscan
import umap
from tqdm.notebook import trange
from hyperopt import fmin, tpe, STATUS_OK, space_eval, Trials
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
        """
        ChatIntents initialization

        Arguments:
            message_embeddings: numpy array of document embeddings created
                                using desired model for class instance
            name: string short name (no spaces) to be used for columnn
                  identifiers
            best_params: UMAP + HDBSCAN hyperparameters associated with
                         the best performance after performing bayesian search
                         using 'bayesian_search' method
            best_clusters: HDBSCAN clusters and labels associated with
                           the best performance after performing baysiean
                           search using 'bayesian_search' method
            trials: hyperopt trials saved from bayesian search using
                    'bayesian_search' method
        """

    def generate_clusters(self,
                          n_neighbors,
                          n_components,
                          min_cluster_size,
                          min_samples=None,
                          random_state=None):
        """
        Generate HDBSCAN clusters from UMAP embeddings of instance message
        embeddings

        Arguments:
            n_neighbors: float, UMAP n_neighbors parameter representing the
                         size of local neighborhood (in terms of number of
                         neighboring sample points) used
            n_components: int, UMAP n_components parameter representing
                          dimension of the space to embed into
            min_cluster_size: int, HDBSCAN parameter minimum size of clusters
            min_samples: int, HDBSCAN parameter representing the number of
                         samples in a neighbourhood for a point to be
                         considered a core point
            random_state: int, random seed to use in UMAP process

        Returns:
            clusters: HDBSCAN clustering object storing results of fit
                      to instance message embeddings

        """

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
        Returns the label count and cost of a given clustering

        Arguments:
            clusters: HDBSCAN clustering object
            prob_threshold: float, probability threshold to use for deciding
                            what cluster labels are considered low confidence

        Returns:
            label_count: int, number of unique cluster labels, including noise
            cost: float, fraction of data points whose cluster assignment has
                  a probability below cutoff threshold
        """

        cluster_labels = clusters.labels_
        label_count = len(np.unique(cluster_labels))
        total_num = len(clusters.labels_)
        cost = (np.count_nonzero(clusters.probabilities_ < prob_threshold)
                / total_num)

        return label_count, cost

    def random_search(self, space, num_evals):
        """
        Randomly search parameter space of clustering pipeline

        Arguments:
            space: dict, contains keys for 'n_neighbors', 'n_components',
                   'min_cluster_size' and 'min_samples' and values with
                   corresponding lists or ranges of parameters to search
            num_evals: int, number of random parameter combinations to try

        Returns:
            df_result: pandas dataframe containing info on each evaluation
                       performed, including run_id, parameters used, label
                       count, and cost
        """

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

            label_count, cost = self.score_clusters(clusters,
                                                    prob_threshold=0.05)

            results.append([i, n_neighbors, n_components, min_cluster_size,
                            min_samples, label_count, cost])

        df_result = pd.DataFrame(results,
                                 columns=['run_id', 'n_neighbors',
                                          'n_components', 'min_cluster_size',
                                          'min_samples', 'label_count', 'cost']
                                 )

        return df_result.sort_values(by='cost')

    def _objective(self, params, label_lower, label_upper):
        """
        Objective function for hyperopt to minimize

        Arguments:
            params: dict, contains keys for 'n_neighbors', 'n_components',
                   'min_cluster_size', 'min_samples', 'random_state' and
                   their values to use for evaluation
            label_lower: int, lower end of range of number of expected clusters
            label_upper: int, upper end of range of number of expected clusters

        Returns:
            loss: cost function result incorporating penalties for falling
                  outside desired range for number of clusters
            label_count: int, number of unique cluster labels, including noise
            status: string, hypoeropt status

        """

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
        """
        Perform bayesian search on hyperparameter space using hyperopt

        Arguments:
            space: dict, contains keys for 'n_neighbors', 'n_components',
                   'min_cluster_size', 'min_samples', and 'random_state' and
                   values that use built-in hyperopt functions to define
                   search spaces for each
            label_lower: int, lower end of range of number of expected clusters
            label_upper: int, upper end of range of number of expected clusters
            max_evals: int, maximum number of parameter combinations to try

        Saves the following to instance variables:
            best_params: dict, contains keys for 'n_neighbors', 'n_components',
                   'min_cluster_size', 'min_samples', and 'random_state' and
                   values associated with lowest cost scenario tested
            best_clusters: HDBSCAN object associated with lowest cost scenario
                           tested
            trials: hyperopt trials object for search

        """

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
        """
        Reduce dimensionality of best clusters and plot in 2D using instance
        variable result of running bayesian_search

        Arguments:
            n_neighbors: float, UMAP hyperparameter n_neighbors
            min_dist: float, UMAP hyperparameter min_dist for effective
                      minimum distance between embedded points

        """

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
        """
        Return single category of documents with known labels

        Arguments:
            df: pandas dataframe of documents and associated ground truth
                labels
            category_col: str, name of column with document labels
            category: str, single document label of interest

        Returns:
            single_category: pandas dataframe with only documents from a
                             single category of interest

        """

        single_category = (df[df[category_col] == category]
                           .reset_index(drop=True)
                           )

        return single_category

    @staticmethod
    def _most_common(lst, n_words):
        """
        Return most common n words in list of words

        Arguments:
            lst: list of words
            n_words: int, number of top words by frequency to return

        Returns:
            counter.most_common(n_words): a list of the n most common elements
                                          and their counts from the most
                                          common to the least

        """

        counter = collections.Counter(lst)

        return counter.most_common(n_words)

    def _extract_labels(self, category_docs):
        """
        Extract labels from documents in the same cluster by concatenating
        most common verbs, ojects, and nouns

        Argument:
            category_docs: list of documents, all from the same category or
                        clustering

        Returns:
            label: str, group label derived from concatentating most common
                   verb, object, and two most common nouns

        """

        verbs = []
        dobjs = []
        nouns = []
        adjs = []

        verb = ''
        dobj = ''
        noun1 = ''
        noun2 = ''

        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading language model for the spaCy dependency parser\n"
                  "(only required the first time this is run)\n")
            from spacy.cli import download
            download("en_core_web_sm")
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

        if len(verbs) > 0:
            verb = self._most_common(verbs, 1)[0][0]

        if len(dobjs) > 0:
            dobj = self._most_common(dobjs, 1)[0][0]

        if len(nouns) > 0:
            noun1 = self._most_common(nouns, 1)[0][0]

        if len(set(nouns)) > 1:
            noun2 = self._most_common(nouns, 2)[1][0]

        words = [verb, dobj]

        # ensures duplicated words aren't included
        for word in [noun1, noun2]:
            if word not in words:
                words.append(word)

        if '' in words:
            words.remove('')

        label = '_'.join(words)

        return label

    def apply_and_summarize_labels(self, df_data):
        """
        Assign groups to original documents and provide group counts

        Arguments:
            df_data: pandas dataframe of original documents of interest to
                     cluster

        Returns:
            df_summary: pandas dataframe with model cluster assignment, number
                        of documents in each cluster and derived labels
            labeled_docs: pandas dataframe with model cluster assignment and
                          associated dervied label applied to each document in
                          original corpus

        """

        # create a dataframe with cluster numbers applied to each doc
        category_col = 'label_' + self.name
        df_clustered = df_data.copy()
        df_clustered[category_col] = self.best_clusters.labels_

        numerical_labels = df_clustered[category_col].unique()

        # create dictionary mapping the numerical category to the generated
        # label
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


def _combine_results(df_ground, model_lst):
    """
    Returns dataframe of all documents and each model's assigned cluster

    Arguments:
        df_ground: dataframe of original documents with associated ground truth
                   labels
        model_lst: list of model ChatIntent instances to include in evaluation

    Returns:
        df_combined: dataframe of all documents with labels from
                     best clusters for each model

    """

    df_combined = df_ground.copy()

    for model in model_lst:
        label_name = 'label_' + model.name
        df_combined[label_name] = model.best_clusters.labels_

    return df_combined


def evaluate_models(df_ground, model_lst):
    """
    Returns a table summarizing each model's performance compared to ground
    truth labels and also labels all docs for each model being evaluated

    Arguments:
        df_ground: dataframe of original documents with associated ground truth
                   labels
        model_lst: list of model ChatIntent instances to include in evaluation

    Returns:
        df_evaluation: dataframe with each row including a model name and
                       calculated ARI and NMI
        df_combined: dataframe of all documents with labels from
                     best clusters for each model

    """

    df_combined = _combine_results(df_ground, model_lst)

    summary = []

    for model in model_lst:
        ground_labels = df_combined['category'].values
        clustered_labels = df_combined['label_' + model.name].values

        ari = np.round(adjusted_rand_score(ground_labels,
                                           clustered_labels), 3)
        nmi = np.round(normalized_mutual_info_score(ground_labels,
                                                    clustered_labels), 3)
        loss = model.trials.best_trial['result']['loss']
        label_count = model.trials.best_trial['result']['label_count']
        n_neighbors = model.best_params['n_neighbors']
        n_components = model.best_params['n_components']
        min_cluster_size = model.best_params['min_cluster_size']
        random_state = model.best_params['random_state']
        summary.append([model.name, ari, nmi, loss, label_count, n_neighbors,
                        n_components, min_cluster_size, random_state])

    df_evaluation = pd.DataFrame(summary, columns=['Model', 'ARI', 'NMI', 
                                                   'loss', 'label_count',
                                                   'n_neighbors', 
                                                   'n_components',
                                                   'min_cluster_size',
                                                   'random_state'])

    return df_evaluation.sort_values(by='NMI', ascending=False), df_combined


def top_cluster_category(df_clusters, df_ground, key, df_summary):
    """
    Returns a dataframe comparing a single model's results to ground truth
    label to evalute cluster compositions and derived label relative to labels
    and counts of most commmon ground truth category

    Arguments:
        df_clusters: pandas dataframe with model cluster assignment and
                     associated dervied label applied to each document in
                     original corpus (labeled_docs result from
                     `apply_and_summarize_labels`)
        df_ground: dataframe of original documents with associated ground truth
                   labels
        key: str, column name to use for joining tables corresponding to
             document text
        df_summary: pandas dataframe with model cluster assignment, number
                    of documents in each cluster and derived labels
                    (df_summary result from `apply_and_summarize_labels`)

    Returns:
        df_result: pandas dataframe with each row containing information on
                   each cluster identified by this model, including count,
                   extracted label, most represented ground truth label name,
                   count and percentage of that group
    """

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
