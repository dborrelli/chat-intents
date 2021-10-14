import pytest
import numpy as np
import pandas as pd
from hyperopt import hp
from sentence_transformers import SentenceTransformer

import chatintents
from chatintents.ChatIntents import ChatIntents


# randomly sample 200 intents from bank77 dataset
url = 'https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv'
data = pd.read_csv(url, sep=',', header=0).sample(200, random_state=42)


@pytest.fixture
def embeddings_st1():
    '''Returns embeddings of simple corpus of docs'''

    docs = list(data['text'])
    model = SentenceTransformer('all-distilroberta-v1')
    embeddings = model.encode(docs)
    return embeddings


@pytest.fixture
def chat_intents_st1(embeddings_st1):
    '''Returns an ChatIntents instance using sentence transformer embeddings of
    simple example docs and with name st1
    '''
    return ChatIntents(message_embeddings=embeddings_st1, name="st1")


@pytest.fixture
def opt_clusters_st1(chat_intents_st1):
    '''Returns ChatIntents instance after running bayesian optimization
    to find best clusters'''

    hspace = {
             "n_neighbors": hp.choice('n_neighbors', range(3, 16)),
             "n_components": hp.choice('n_components', range(3, 16)),
             "min_cluster_size": hp.choice('min_cluster_size', range(2, 16)),
             "min_samples": None,
             "random_state": 42
             }

    label_lower = 30
    label_upper = 100
    max_evals = 10

    chat_intents_st1.bayesian_search(space=hspace,
                                     label_lower=label_lower,
                                     label_upper=label_upper,
                                     max_evals=max_evals)
    return chat_intents_st1


@pytest.fixture
def opt_clusters_st2():
    '''Returns ChatIntents instance after running bayesian optimization
    to find best clusters'''

    docs = list(data['text'])
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings_st2 = model.encode(docs)

    chat_intents_st2 = ChatIntents(message_embeddings=embeddings_st2,
                                   name="st2")

    hspace = {
             "n_neighbors": hp.choice('n_neighbors', range(3, 16)),
             "n_components": hp.choice('n_components', range(3, 16)),
             "min_cluster_size": hp.choice('min_cluster_size', range(2, 16)),
             "min_samples": None,
             "random_state": 42
             }

    label_lower = 30
    label_upper = 100
    max_evals = 10

    chat_intents_st2.bayesian_search(space=hspace,
                                     label_lower=label_lower,
                                     label_upper=label_upper,
                                     max_evals=max_evals)
    return chat_intents_st2


def test_setting_embeddings(chat_intents_st1):
    assert chat_intents_st1.message_embeddings.shape[0] == 200
    assert chat_intents_st1.message_embeddings.shape[1] > 5


def test_setting_name(chat_intents_st1):
    assert chat_intents_st1.name == "st1"


def test_generate_clusters(chat_intents_st1):
    clusters = chat_intents_st1.generate_clusters(n_neighbors=15,
                                                  n_components=10,
                                                  min_cluster_size=5,
                                                  min_samples=None,
                                                  random_state=42)
    assert len(clusters.labels_) == 200


def test_score_clusters(chat_intents_st1):
    clusters = chat_intents_st1.generate_clusters(n_neighbors=15,
                                                  n_components=10,
                                                  min_cluster_size=5,
                                                  min_samples=None,
                                                  random_state=42)

    label_count, cost = chat_intents_st1.score_clusters(clusters,
                                                        prob_threshold=0.05)
    print(label_count)
    print(cost)
    assert label_count == 11
    assert cost == 0.255


def test_random_search(chat_intents_st1):
    space = {
        "n_neighbors": range(12, 16),
        "n_components": range(3, 7),
        "min_cluster_size": range(2, 15),
        "min_samples": range(2, 15)
    }

    df_random = chat_intents_st1.random_search(space, 5)

    assert len(df_random) == 5
    assert isinstance(df_random, pd.core.frame.DataFrame)


def test_bayesian_search(opt_clusters_st1):
    assert opt_clusters_st1.best_params is not None
    assert set(opt_clusters_st1.best_params.keys()) == set(['n_neighbors',
                                                            'n_components',
                                                            'min_cluster_size',
                                                            'min_samples',
                                                            'random_state'])

    assert opt_clusters_st1.best_clusters is not None
    assert len(opt_clusters_st1.best_clusters.labels_) == 200

    assert opt_clusters_st1.trials is not None
    assert len(opt_clusters_st1.trials.trials) == 10


def test_plot_best_clusters(opt_clusters_st1):
    opt_clusters_st1.plot_best_clusters()


def test_apply_and_summarize_labels(opt_clusters_st1):
    df_summary, labeled_docs = opt_clusters_st1.apply_and_summarize_labels(data[['text']])
    assert len(df_summary) == len(np.unique(opt_clusters_st1.best_clusters.labels_))
    assert df_summary['count'].sum() == 200
    assert all([isinstance(label, str) for label in list(df_summary['label'])])

    assert len(labeled_docs) == 200


def test_evaluate_models(opt_clusters_st1, opt_clusters_st2):
    models = [opt_clusters_st1, opt_clusters_st2]
    df_comparison, labeled_docs_all_models = chatintents.evaluate_models(data[['text', 'category']],
                                                                         models)

    assert len(df_comparison) == len(models)
    assert isinstance(df_comparison, pd.core.frame.DataFrame)

    assert len(labeled_docs_all_models) == 200
    assert all([col in labeled_docs_all_models.columns for col in ['label_st1',
                                                                   'label_st2']])


def test_top_cluster_category(opt_clusters_st1):
    df_summary, labeled_docs = opt_clusters_st1.apply_and_summarize_labels(data[['text']])

    df_result = chatintents.top_cluster_category(labeled_docs,
                                                 data[['text', 'category']],
                                                 'text',
                                                 df_summary)

    assert len(df_summary) == len(np.unique(opt_clusters_st1.best_clusters.labels_))
    assert df_summary['count'].sum() == 200
    assert len(df_result.columns) == 6
