from codekicker.classifier import (classify, classify_sentence,
                                   get_frequency_in_sentence,
                                   sum_target_feature_frequencies)
from codekicker.preprocesser import extract_features_and_vocabulary
from pytest import fixture


@fixture
def input_sentences():
    return ['Eine Fehlermeldung „Sie benötigen benötigen Adminrechte“ poppt auf.',
            'Der Versendeladebalken stoppt.']


@fixture
def stemmed_target_features():
    return [['poppt', 'benot'], ['der', 'stoppt']]


@fixture
def features(input_sentences):
    return extract_features_and_vocabulary(input_sentences)[0].toarray()


@fixture
def vocabulary(input_sentences):
    return extract_features_and_vocabulary(input_sentences)[1]


def test_get_frequency_in_sentence(features, vocabulary):
    assert 1 == get_frequency_in_sentence('poppt', features[0], vocabulary)


def test_sum_target_feature_frequencies(stemmed_target_features, vocabulary, features):
    assert 3 == sum_target_feature_frequencies(stemmed_target_features[0], features[0], vocabulary)
    assert 0 == sum_target_feature_frequencies(stemmed_target_features[1], features[0], vocabulary)


def test_classify_sentence(stemmed_target_features, features, vocabulary):
    assert 0 == classify_sentence(stemmed_target_features, features[0], vocabulary)
    assert 1 == classify_sentence(stemmed_target_features, features[1], vocabulary)


def test_classify(stemmed_target_features, features, vocabulary):
    assert [0, 1] == classify(stemmed_target_features, features, vocabulary)
