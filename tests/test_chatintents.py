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
def create_embeddings():
    '''Returns embeddings of simple corpus of docs'''

    docs = list(data['text'])
    model = SentenceTransformer('all-distilroberta-v1')
    embeddings = model.encode(docs)
    return embeddings


@pytest.fixture
def chat_intents(create_embeddings):
    '''Returns an ChatIntents instance using sentence transformer embeddings of
    simple example docs and with name st1
    '''
    return ChatIntents(message_embeddings=create_embeddings, name="st1")


@pytest.fixture
def optim_clusters(chat_intents):
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

    chat_intents.bayesian_search(space=hspace,
                                 label_lower=label_lower,
                                 label_upper=label_upper,
                                 max_evals=max_evals)
    return chat_intents


def test_setting_embeddings(chat_intents):
    assert chat_intents.message_embeddings.shape[0] == 200
    assert chat_intents.message_embeddings.shape[1] > 5


def test_setting_name(chat_intents):
    assert chat_intents.name == "st1"


def test_generate_clusters(chat_intents):
    clusters = chat_intents.generate_clusters(n_neighbors=15,
                                              n_components=10,
                                              min_cluster_size=5,
                                              min_samples=None,
                                              random_state=42)
    assert len(clusters.labels_) == 200


def test_score_clusters(chat_intents):
    clusters = chat_intents.generate_clusters(n_neighbors=15,
                                              n_components=10,
                                              min_cluster_size=5,
                                              min_samples=None,
                                              random_state=42)

    label_count, cost = chat_intents.score_clusters(clusters,
                                                    prob_threshold=0.05)
    print(label_count)
    print(cost)
    assert label_count == 11
    assert cost == 0.255


def test_random_search(chat_intents):
    space = {
        "n_neighbors": range(12, 16),
        "n_components": range(3, 7),
        "min_cluster_size": range(2, 15),
        "min_samples": range(2, 15)
    }

    df_random = chat_intents.random_search(space, 5)

    assert len(df_random) == 5
    assert isinstance(df_random, pd.core.frame.DataFrame)


def test_bayesian_search(optim_clusters):
    assert optim_clusters.best_params is not None
    assert set(optim_clusters.best_params.keys()) == set(['n_neighbors',
                                                          'n_components',
                                                          'min_cluster_size',
                                                          'min_samples',
                                                          'random_state'])

    assert optim_clusters.best_clusters is not None
    assert len(optim_clusters.best_clusters.labels_) == 200

    assert optim_clusters.trials is not None
    assert len(optim_clusters.trials.trials) == 10


def test_plot_best_clusters(optim_clusters):
    optim_clusters.plot_best_clusters()


def test_apply_and_summarize_labels(optim_clusters):
    df_summary, labeled_docs = optim_clusters.apply_and_summarize_labels(data[['text']])
    assert len(df_summary) == len(np.unique(optim_clusters.best_clusters.labels_))
    assert df_summary['count'].sum() == 200
    assert all([isinstance(label, str) for label in list(df_summary['label'])])

    assert len(labeled_docs) == 200


def test_evaluate_models(optim_clusters):
    models = [optim_clusters, model_st2]
    df_comparison, labeled_docs_all_models = chatintents.evaluate_models(data[['text', 'category']],
                                                                         models)

